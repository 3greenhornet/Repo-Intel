#!/usr/bin/env python3
"""Auto-generate .repointel/architecture.yml from a repository structure with preset templates."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEMPLATE_HEADER = """# RepoIntel architecture intent (Reflexion Model)
# Declare expected component boundaries and dependency flow.
# RepoIntel will flag dependencies that violate these architectural constraints.

intent:
  status: {status} # prototype | mature
  goal: "{goal}"
  preset: "{preset}"

components:
{components}

dependencies:
  allowed:
{allowed}
  forbidden:
{forbidden}
"""

PRESETS = {
    "layered": {
        "components": {
            "presentation": {"paths": ["**/api/**", "**/views/**", "**/controllers/**", "**/routers/**", "**/web/**"]},
            "domain": {"paths": ["**/core/**", "**/domain/**", "**/models/**", "**/services/**"]},
            "infrastructure": {"paths": ["**/storage/**", "**/db/**", "**/database/**", "**/clients/**", "**/adapters/**"]},
        },
        "allowed": [
            {"from": "presentation", "to": "domain"},
            {"from": "presentation", "to": "infrastructure"},
            {"from": "infrastructure", "to": "domain"},
        ],
        "forbidden": [
            {"from": "domain", "to": "presentation"},
            {"from": "domain", "to": "infrastructure"},
        ],
    },
    "hexagonal": {
        "components": {
            "inbound_adapters": {"paths": ["**/api/**", "**/controllers/**", "**/graphql/**", "**/cli/**"]},
            "application": {"paths": ["**/services/**", "**/use_cases/**", "**/commands/**"]},
            "domain": {"paths": ["**/domain/**", "**/models/**", "**/entities/**"]},
            "outbound_adapters": {"paths": ["**/db/**", "**/gateways/**", "**/storage/**", "**/http_clients/**"]},
        },
        "allowed": [
            {"from": "inbound_adapters", "to": "application"},
            {"from": "application", "to": "domain"},
            {"from": "outbound_adapters", "to": "domain"},
        ],
        "forbidden": [
            {"from": "domain", "to": "inbound_adapters"},
            {"from": "domain", "to": "outbound_adapters"},
            {"from": "domain", "to": "application"},
        ],
    },
}


def build_from_preset(preset_name: str, repo: Path, status: str = "prototype") -> str:
    preset = PRESETS[preset_name]
    comp_lines = []
    for name, spec in preset["components"].items():
        comp_lines.append(f"  {name}:")
        comp_lines.append(f"    paths: {spec['paths']}")
    components_str = "\n".join(comp_lines)

    allowed_str = "\n".join(
        f"    - {{ from: {p['from']}, to: {p['to']} }}" for p in preset["allowed"]
    )
    forbidden_str = "\n".join(
        f"    - {{ from: {p['from']}, to: {p['to']} }}" for p in preset["forbidden"]
    )

    return TEMPLATE_HEADER.format(
        status=status,
        goal=f"Enforce {preset_name.capitalize()} Architecture boundaries on {repo.name}",
        preset=preset_name,
        components=components_str,
        allowed=allowed_str,
        forbidden=forbidden_str,
    )


def build_from_directory_layout(repo: Path, status: str = "prototype") -> str:
    top_dirs = sorted([
        d.name for d in repo.iterdir()
        if d.is_dir() and not d.name.startswith((".", "__"))
        and d.name not in ("node_modules", "venv", ".venv", "dist", "build", "tests", "test")
    ])

    if not top_dirs:
        top_dirs = ["app"]

    comp_lines = [f"  {name}:\n    paths: [\"{name}/**\"]" for name in top_dirs]
    components_str = "\n".join(comp_lines)

    allowed_pairs = []
    for i, src in enumerate(top_dirs):
        for dst in top_dirs[i + 1:]:
            allowed_pairs.append(f"    - {{ from: {src}, to: {dst} }}")
            allowed_pairs.append(f"    - {{ from: {dst}, to: {src} }}")
    allowed_str = "\n".join(allowed_pairs) if allowed_pairs else "    []"

    forbidden_pairs = []
    if len(top_dirs) > 1:
        forbidden_pairs.append(f"    # - {{ from: {top_dirs[-1]}, to: {top_dirs[0]} }} # uncomment to forbid")
    forbidden_str = "\n".join(forbidden_pairs) if forbidden_pairs else "    []"

    return TEMPLATE_HEADER.format(
        status=status,
        goal=f"Auto-generated module boundaries for {repo.name}",
        preset="modular",
        components=components_str,
        allowed=allowed_str,
        forbidden=forbidden_str,
    )


def main(repo_path: str, preset: str = "auto", status: str = "prototype", force: bool = False) -> int:
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        print(f"ERROR: not a directory: {repo}", file=sys.stderr)
        return 1

    out_dir = repo / ".repointel"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "architecture.yml"

    if out_file.exists() and not force:
        print(f"EXISTS: {out_file} — not overwriting (use --force to overwrite)")
        return 0

    if preset in PRESETS:
        content = build_from_preset(preset, repo, status=status)
    else:
        content = build_from_directory_layout(repo, status=status)

    out_file.write_text(content, encoding="utf-8")
    print(f"WROTE: {out_file} (preset: {preset}, status: {status})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate .repointel/architecture.yml for intent checking")
    parser.add_argument("repo", help="Path to repository")
    parser.add_argument("--preset", choices=["auto", "layered", "hexagonal"], default="auto", help="Architecture template")
    parser.add_argument("--status", choices=["prototype", "mature"], default="prototype", help="Reflexion strictness status")
    parser.add_argument("--force", action="store_true", help="Overwrite existing architecture.yml")
    args = parser.parse_args()
    raise SystemExit(main(args.repo, preset=args.preset, status=args.status, force=args.force))

