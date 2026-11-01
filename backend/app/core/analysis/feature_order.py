"""Canonical feature order for the quality-prediction models."""
FEATURE_ORDER = [
    "loc", "function_count", "class_count", "cyclomatic_complexity",
    "complexity_per_loc", "token_count", "import_count", "change_frequency",
    "author_count", "avg_function_length", "satd_count", "security_findings",
    "bug_findings", "ast_node_count_proxy",
]
