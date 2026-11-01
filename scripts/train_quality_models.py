#!/usr/bin/env python3
"""Train the software-quality prediction models (§4.2.4 Component D).

Thin wrapper around `app.core.analysis.model_trainer` so `make train` and CI
keep working from the repository root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.analysis.model_trainer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
