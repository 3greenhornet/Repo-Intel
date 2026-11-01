"""Quality-model trainer (§4.2.4 Component D) — backend module.

FIX-023: model artifacts are never committed; they are trained deterministically
(SEED=42) at build time (Dockerfile) or on demand (`make train`). Living inside
the backend package lets the Docker build context (./backend) run it directly.
"""
from __future__ import annotations

import pickle

import numpy as np

from app.config import get_settings
from app.core.analysis.feature_order import FEATURE_ORDER

MODEL_VERSION = "v1.0"
SEED = 42


def synth_dataset(n: int = 4000, seed: int = SEED) -> tuple[np.ndarray, ...]:
    """Generate labeled feature rows following the §4.2.4 feature semantics."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(FEATURE_ORDER)), dtype=np.float32)
    defect = np.zeros(n, dtype=int)
    smell = np.zeros(n, dtype=int)  # 0 none, 1 complex_function, 2 god_file, 3 long_parameter_list
    td = np.zeros(n, dtype=np.float32)

    for i in range(n):
        loc = float(rng.integers(5, 900))
        function_count = float(rng.integers(0, 40))
        class_count = float(rng.integers(0, 8))
        complexity = float(np.clip(rng.gamma(2.0, 6.0, 1)[0], 1, 300))
        complexity_per_loc = complexity / max(loc, 1)
        token_count = loc * rng.uniform(3, 8)
        import_count = float(rng.integers(0, 25))
        change_frequency = float(rng.poisson(2.5))
        author_count = float(rng.integers(1, 6))
        avg_function_length = loc / max(function_count, 1)
        satd_count = float(rng.poisson(0.8))
        security_findings = float(rng.poisson(0.4))
        bug_findings = float(rng.poisson(0.6))

        row = {
            "loc": loc, "function_count": function_count, "class_count": class_count,
            "cyclomatic_complexity": complexity, "complexity_per_loc": complexity_per_loc,
            "token_count": token_count, "import_count": import_count,
            "change_frequency": change_frequency, "author_count": author_count,
            "avg_function_length": avg_function_length, "satd_count": satd_count,
            "security_findings": security_findings, "bug_findings": bug_findings,
            "ast_node_count_proxy": loc,
        }
        X[i] = [row[k] for k in FEATURE_ORDER]

        # labels follow the §4.2.4 / built-in-analyzer semantics
        p_defect = (
            0.35 * min(complexity / 80, 1)
            + 0.20 * min(change_frequency / 15, 1)
            + 0.15 * min(satd_count / 5, 1)
            + 0.15 * min(security_findings / 3, 1)
            + 0.15 * min(bug_findings / 6, 1)
        )
        defect[i] = 1 if rng.random() < p_defect else 0

        if complexity > 60:
            smell[i] = 1
        elif loc > 500:
            smell[i] = 2
        elif avg_function_length > 80:
            smell[i] = 3
        else:
            smell[i] = 0

        td[i] = np.clip(
            0.4 * min(satd_count / 5, 1)
            + 0.3 * min(avg_function_length / 100, 1)
            + 0.3 * min(complexity_per_loc / 0.5, 1)
            + rng.normal(0, 0.03), 0.0, 1.0)

    return X, defect, smell, td


def main() -> int:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier

    settings = get_settings()
    out_dir = settings.models_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X, defect, smell, td = synth_dataset()
    split = int(len(X) * 0.8)
    Xtr, Xte = X[:split], X[split:]
    print(f"dataset: {len(X)} rows, {len(FEATURE_ORDER)} features")

    defect_clf = GradientBoostingClassifier(random_state=SEED, n_estimators=120)
    defect_clf.fit(Xtr, defect[:split])
    acc = defect_clf.score(Xte, defect[split:])
    print(f"defect_predictor  accuracy: {acc:.3f}")

    smell_clf = RandomForestClassifier(random_state=SEED, n_estimators=150)
    smell_clf.fit(Xtr, smell[:split])
    acc = smell_clf.score(Xte, smell[split:])
    print(f"smell_predictor   accuracy: {acc:.3f}")

    td_reg = GradientBoostingRegressor(random_state=SEED, n_estimators=120)
    td_reg.fit(Xtr, td[:split])
    r2 = td_reg.score(Xte, td[split:])
    print(f"td_regressor      R²:      {r2:.3f}")

    for name, model in [
        ("defect_predictor", defect_clf),
        ("smell_predictor", smell_clf),
        ("td_regressor", td_reg),
    ]:
        path = out_dir / f"{name}_{MODEL_VERSION}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"saved {path}")
    print(f"\nAll models written to {out_dir} (version {MODEL_VERSION})")
    return 0


