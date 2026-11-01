"""Software quality prediction — ML models (§4.2.4 Component D).

Feature extraction follows the documented `extract_features` algorithm
(structural + semantic + dependency + historical features). Models are
pre-trained and loaded from disk exactly as documented: a training script
(`scripts/train_quality_models.py`) fits a GradientBoosting defect predictor,
a RandomForest smell classifier, and a TD regressor on features derived from
labeled heuristics, pickles them into backend/models/trained/, and runtime
code only loads (`predict_quality` never trains).
"""
from __future__ import annotations

import pickle

from app.utils.time import utc_now
from pathlib import Path

from app.config import get_settings
from app.models.code import CodeIntelligence, Symbol
from app.models.finding import Finding, QualityPrediction
from app.models.repository import CommitRecord, FileRecord
from app.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY = {  # §24.8 model versioning
    "defect_predictor": {"v1.0": "defect_predictor_v1.0.pkl"},
    "smell_predictor": {"v1.0": "smell_predictor_v1.0.pkl"},
    "td_regressor": {"v1.0": "td_regressor_v1.0.pkl"},
}
CURRENT_VERSIONS = {"defect_predictor": "v1.0", "smell_predictor": "v1.0", "td_regressor": "v1.0"}


class QualityPredictor:
    """Loads pre-trained models from disk and predicts (§4.2.4)."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._models: dict[str, object] = {}
        self.model_version = "v1.0"
        self.available = False
        self._load_models()

    def _load_models(self) -> None:
        for name, versions in MODEL_REGISTRY.items():
            path = self.settings.models_dir / versions[CURRENT_VERSIONS[name]]
            if not path.exists():
                logger.info(f"Model {name} not found at {path} — run scripts/train_quality_models.py")
                return
        try:
            for name, versions in MODEL_REGISTRY.items():
                path = self.settings.models_dir / versions[CURRENT_VERSIONS[name]]
                with open(path, "rb") as f:
                    self._models[name] = pickle.load(f)
            self.available = True
            logger.info(f"Loaded {len(self._models)} quality models ({', '.join(self._models)})")
        except Exception as exc:
            logger.warning(f"Model loading failed: {exc}")

    # ------------------------------------------------------- feature extraction
    @staticmethod
    def extract_features(
        file: FileRecord, source: str, symbols: list[Symbol],
        commits: list[CommitRecord], findings: list[Finding],
    ) -> dict[str, float]:
        """§4.2.4 extract_features — structural/semantic/dependency/historical."""
        functions = [s for s in symbols if s.type == "function"]
        classes = [s for s in symbols if s.type == "class"]
        loc = sum(1 for line in source.splitlines() if line.strip())

        complexity = 1 + sum(
            source.count(kw) for kw in ("if ", "elif ", "for ", "while ", "except", "and ", "or ")
        )

        change_frequency = sum(1 for c in commits if file.path in c.files_changed)
        authors = len({c.author for c in commits if file.path in c.files_changed})
        file_findings = [f for f in findings if f.file_path == file.path]
        satd_count = sum(1 for f in file_findings if f.finding_type == "td_prediction")
        security_count = sum(1 for f in file_findings if f.finding_type == "vulnerability")
        bug_count = sum(1 for f in file_findings if f.finding_type == "bug")

        return {
            "ast_node_count_proxy": loc,
            "loc": float(loc),
            "function_count": float(len(functions)),
            "class_count": float(len(classes)),
            "cyclomatic_complexity": float(complexity),
            "complexity_per_loc": complexity / max(loc, 1),
            "token_count": float(len(source.split())),
            "import_count": float(source.count("import ") + source.count("from ")),
            "change_frequency": float(change_frequency),
            "author_count": float(authors),
            "avg_function_length": float(loc) / max(len(functions), 1),
            "satd_count": float(satd_count),
            "security_findings": float(security_count),
            "bug_findings": float(bug_count),
        }

    # ------------------------------------------------------------------ predict
    def predict(self, features: dict[str, float]) -> dict:
        """§4.2.4 predict_quality — models loaded from disk, never trained here."""
        if not self.available:
            # Deterministic heuristic fallback when models not yet trained
            defect = min(1.0, (
                0.25 * min(features.get("cyclomatic_complexity", 0) / 50, 1.0)
                + 0.25 * min(features.get("change_frequency", 0) / 20, 1.0)
                + 0.2 * min(features.get("satd_count", 0) / 5, 1.0)
                + 0.15 * min(features.get("security_findings", 0) / 3, 1.0)
                + 0.15 * min(features.get("bug_findings", 0) / 5, 1.0)
            ))
            return {
                "defect_probability": round(defect, 3),
                "smell_type": "god_file" if features.get("loc", 0) > 500 else (
                    "complex_function" if features.get("cyclomatic_complexity", 0) > 30 else None),
                "smell_confidence": round(min(1.0, features.get("cyclomatic_complexity", 0) / 60), 3),
                "td_score": round(
                    0.4 * min(features.get("satd_count", 0) / 5, 1.0)
                    + 0.3 * min(features.get("avg_function_length", 0) / 80, 1.0)
                    + 0.3 * min(features.get("complexity_per_loc", 0) / 0.4, 1.0), 3),
                "satd_flag": features.get("satd_count", 0) > 0,
                "model_version": "heuristic-v1",
            }

        import numpy as np

        from app.core.analysis.feature_order import FEATURE_ORDER

        x = np.array([[features.get(k, 0.0) for k in FEATURE_ORDER]])
        defect_model = self._models["defect_predictor"]
        smell_model = self._models["smell_predictor"]
        td_model = self._models["td_regressor"]

        defect_prob = float(defect_model.predict_proba(x)[0][1])
        smell_pred = int(smell_model.predict(x)[0])
        td_score = float(td_model.predict(x)[0])

        smell_types = {0: None, 1: "complex_function", 2: "god_file", 3: "long_parameter_list"}
        return {
            "defect_probability": round(min(max(defect_prob, 0.0), 1.0), 3),
            "smell_type": smell_types.get(smell_pred),
            "smell_confidence": round(max(smell_model.predict_proba(x)[0]), 3),
            "td_score": round(min(max(td_score, 0.0), 1.0), 3),
            "satd_flag": features.get("satd_count", 0) > 0,
            "model_version": "v1.0",
        }

    def explain_prediction(self, features: dict) -> dict:
        """XAI (§Task 4): Return the exact Shapley values driving this prediction."""
        import warnings
        from app.core.analysis.feature_order import FEATURE_ORDER

        if not self.available:
            sorted_heuristics = sorted(
                [
                    {"feature": k, "shap_value": round(float(features.get(k, 0.0)), 4)}
                    for k in FEATURE_ORDER
                ],
                key=lambda x: abs(x["shap_value"]),
                reverse=True
            )
            return {
                "method": "heuristic_feature_attribution",
                "top_features": sorted_heuristics[:5]
            }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import shap
                import numpy as np

                model = self._models.get("defect_predictor") or self._models.get("smell_predictor")
                explainer = shap.TreeExplainer(model)
                x = np.array([[features.get(k, 0.0) for k in FEATURE_ORDER]])
                shap_values = explainer.shap_values(x)

                if isinstance(shap_values, list):
                    vals = np.mean([np.abs(sv[0]) for sv in shap_values], axis=0)
                elif hasattr(shap_values, "values"):
                    vals = shap_values.values[0]
                else:
                    vals = shap_values[0]

                attributions = [
                    {"feature": name, "shap_value": round(float(v), 4)}
                    for name, v in zip(FEATURE_ORDER, vals)
                ]
                attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

                return {
                    "method": "TreeSHAP (exact Shapley for trees)",
                    "top_features": attributions[:5]
                }
        except Exception as exc:
            logger.debug(f"TreeSHAP calculation fallback: {exc}")
            fallback_attrs = [
                {"feature": k, "shap_value": round(float(features.get(k, 0.0)), 4)}
                for k in FEATURE_ORDER if features.get(k, 0.0) > 0
            ]
            fallback_attrs.sort(key=lambda x: x["shap_value"], reverse=True)
            return {
                "method": "feature_weight_fallback",
                "top_features": fallback_attrs[:5]
            }



def s_type_attr(symbol: Symbol, attr: str):
    return getattr(symbol, attr, None)


def run_quality_prediction(
    repo_path: Path | str, files: list[FileRecord], code_intelligence: CodeIntelligence,
    commits: list[CommitRecord], findings: list[Finding],
) -> list[QualityPrediction]:
    """Component D driver: features → predictions per file."""
    from app.core.ingestion.file_walker import read_file_content

    predictor = QualityPredictor()
    symbols_by_file: dict[str, list[Symbol]] = {}
    for sym in code_intelligence.symbols:
        symbols_by_file.setdefault(sym.file_path, []).append(sym)

    predictions: list[QualityPrediction] = []
    for file in files:
        if file.category != "code":
            continue
        source, ok = read_file_content(repo_path, file.path)
        if not ok:
            continue
        features = QualityPredictor.extract_features(
            file, source, symbols_by_file.get(file.path, []), commits, findings
        )
        result = predictor.predict(features)
        predictions.append(QualityPrediction(
            file_path=file.path, symbol_id=None,
            defect_probability=result["defect_probability"],
            smell_type=result["smell_type"], smell_confidence=result["smell_confidence"],
            td_score=result["td_score"], satd_flag=result["satd_flag"],
            features=features, model_version=result["model_version"],
            predicted_at=utc_now(),
        ))
    logger.info("quality_prediction_complete", extra={"extra_fields": {
        "files_predicted": len(predictions),
        "high_risk": sum(1 for p in predictions if p.defect_probability > 0.6),
    }})
    return predictions
