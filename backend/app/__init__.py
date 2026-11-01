"""RepoIntel AI — backend package.

FIX-020: single source of truth for the minimum Python version. The range is
3.12–3.13 (see pyproject.toml `requires-python`) — 3.12 is the minimum the
codebase supports, 3.13 is what CI/Docker use.
"""
import sys

if sys.version_info < (3, 12):
    raise RuntimeError(
        f"RepoIntel AI requires Python 3.12 or newer. "
        f"Running {sys.version_info.major}.{sys.version_info.minor}.")
