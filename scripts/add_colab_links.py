#!/usr/bin/env python3
"""Add or refresh the leading Colab badge in notebooks.

Usage:
    python scripts/add_colab_links.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parent.parent
COLAB_REPO_URL = "https://colab.research.google.com/github/Gan4x4/cv/blob/main"
COLAB_BADGE_URL = "https://colab.research.google.com/assets/colab-badge.svg"
COLAB_NOTEBOOK_URL_RE = re.compile(
    r"https://colab\.research\.google\.com/github/.*?\.ipynb"
)


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


def is_colab_badge_cell(cell: dict[str, Any]) -> bool:
    return "colab-badge.svg" in source_text(cell.get("source")).lower()


def colab_url(path: Path) -> str:
    relative_path = quote(path.relative_to(REPO_ROOT).as_posix(), safe="/")
    return f"{COLAB_REPO_URL}/{relative_path}"


def colab_cell(path: Path) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f'<a href="{colab_url(path)}">\n',
            f'  <img src="{COLAB_BADGE_URL}" alt="Open In Colab"/>\n',
            "</a>\n",
        ],
    }


def refresh_colab_urls(cell: dict[str, Any], path: Path) -> bool:
    source = cell.get("source")
    expected_url = colab_url(path)

    def refresh(part: Any) -> str:
        return COLAB_NOTEBOOK_URL_RE.sub(expected_url, str(part))

    if isinstance(source, list):
        refreshed = [refresh(part) for part in source]
        if refreshed == source:
            return False
        cell["source"] = refreshed
        return True

    refreshed = refresh(source or "")
    if refreshed == source:
        return False
    cell["source"] = refreshed
    return True


def sync_colab_link(notebook: dict[str, Any], path: Path) -> bool:
    cells = notebook.setdefault("cells", [])
    expected_cell = colab_cell(path)

    if cells and is_colab_badge_cell(cells[0]):
        if cells[0].get("source") == expected_cell["source"]:
            return False
        if refresh_colab_urls(cells[0], path):
            return True
        if colab_url(path) in source_text(cells[0].get("source")):
            return False
        cells[0] = expected_cell
    elif has_colab_link(notebook):
        return False
    else:
        cells.insert(0, expected_cell)

    return True


def add_colab_link(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if not sync_colab_link(notebook, path):
        return False

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
