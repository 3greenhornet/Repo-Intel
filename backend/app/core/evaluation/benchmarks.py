"""Component 9A-bis: Benchmark harness (§14.2).

Documented external benchmarks (SWE-bench, RepoQA, BigVul, Juliet) require
multi-GB downloads and external execution harnesses — loaders are provided
and activate when `datasets`/network are available. The built-in
``repointel-local`` benchmark provides labeled questions over any ingested
repository so evaluation runs immediately.
"""
from __future__ import annotations

from pathlib import Path

# Labeled benchmark over the demo fixture repository. Each entry carries
# ground-truth relevant files/symbols for retrieval metrics (§14.3.1).
DEMO_BENCHMARK = [
    {
        "question": "What security vulnerabilities exist in the authentication code?",
        "relevant_files": ["demo_shop/auth.py"],
        "relevant_symbols": ["validate_credentials", "hash_password", "evaluate_rule"],
    },
    {
        "question": "Why is login slow when the database is under load?",
        "relevant_files": ["demo_shop/auth.py", "issue:1"],
        "relevant_symbols": ["login", "validate_credentials"],
    },
    {
        "question": "What happens if I change validate_credentials?",
        "relevant_files": ["demo_shop/auth.py", "demo_shop/api.py"],
        "relevant_symbols": ["validate_credentials", "login", "handle_login"],
    },
    {
        "question": "Who calls the login function?",
        "relevant_files": ["demo_shop/api.py", "demo_shop/auth.py"],
        "relevant_symbols": ["login", "handle_login"],
    },
    {
        "question": "How does the shopping cart compute its total?",
        "relevant_files": ["demo_shop/cart.py", "demo_shop/utils.py"],
        "relevant_symbols": ["Cart", "calculate_total", "total"],
    },
    {
        "question": "How are passwords stored and compared?",
        "relevant_files": ["demo_shop/auth.py"],
        "relevant_symbols": ["hash_password", "validate_credentials"],
    },
    {
        "question": "What is the architecture of this project?",
        "relevant_files": ["docs/ARCHITECTURE.md", "README.md"],
        "relevant_symbols": [],
    },
    {
        "question": "Which files are technical debt hotspots?",
        "relevant_files": ["demo_shop/auth.py"],
        "relevant_symbols": [],
    },
]


def load_swebench_lite(max_items: int = 20) -> list[dict]:
    """§14.2.1 SWE-bench (understanding/localization subset, not resolution)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    except Exception:
        return []
    tasks = []
    for item in list(ds)[:max_items]:
        tasks.append({
            "question": f"[{item['repo']}] {item['problem_statement'][:1500]}",
            "relevant_files": [p.split(" ")[-1] for p in item.get("patch", "").split("diff --git") if p.strip()],
            "relevant_symbols": [],
            "meta": {"instance_id": item.get("instance_id"), "repo": item["repo"]},
        })
    return tasks


def load_repoqa(max_items: int = 20) -> list[dict]:
    """§14.2.2 RepoQA — hit rate on ground-truth function retrieval."""
    repoqa_dir = Path("data/repos/repoqa/data")
    if not repoqa_dir.exists():
        return []
    import json

    tasks = []
    for lang_file in sorted(repoqa_dir.glob("*.jsonl")):
        with open(lang_file) as f:
            for line in f:
                if len(tasks) >= max_items:
                    break
                item = json.loads(line)
                tasks.append({
                    "question": item.get("query", ""),
                    "relevant_files": [],
                    "relevant_symbols": [item.get("ground_truth_function", "")],
                })
    return tasks


def get_benchmark(name: str, max_items: int = 20) -> list[dict]:
    if name == "repointel-local":
        return DEMO_BENCHMARK
    if name == "swe-bench-lite":
        return load_swebench_lite(max_items)
    if name == "repoqa":
        return load_repoqa(max_items)
    raise ValueError(f"Unknown benchmark: {name}")
