#!/usr/bin/env python3
"""Add a Colab badge to notebooks that do not already have one.

Usage:
    python scripts/add_colab_links.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parent.parent
COLAB_REPO_URL = "https://colab.research.google.com/github/Gan4x4/cv/blob/main"
COLAB_BADGE_URL = "https://colab.research.google.com/assets/colab-badge.svg"


def notebook_paths() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )


def source_text(source: Any) -> str:
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source or "")


def has_colab_link(notebook: dict[str, Any]) -> bool:
    for cell in notebook.get("cells", []):
        text = source_text(cell.get("source")).lower()
        if "colab-badge.svg" in text or "open in colab" in text:
            return True
    return False


def colab_cell(path: Path) -> dict[str, Any]:
    relative_path = quote(path.relative_to(REPO_ROOT).as_posix(), safe="/")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f'<a href="{COLAB_REPO_URL}/{relative_path}">\n',
            f'  <img src="{COLAB_BADGE_URL}" alt="Open In Colab"/>\n',
            "</a>\n",
        ],
    }


def add_colab_link(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if has_colab_link(notebook):
        return False

    notebook.setdefault("cells", []).insert(0, colab_cell(path))
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def main() -> int:
    changed = 0
    paths = notebook_paths()
    for path in paths:
        if add_colab_link(path):
            changed += 1
            print(path.relative_to(REPO_ROOT))

    print(f"updated {changed} of {len(paths)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
