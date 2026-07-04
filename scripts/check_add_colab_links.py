#!/usr/bin/env python3
"""Small self-check for scripts/add_colab_links.py."""

from __future__ import annotations

import copy

import add_colab_links


def main() -> int:
    path = add_colab_links.REPO_ROOT / "OCR/Text_detection_DB-Net_EAST_CRAFT.ipynb"
    stale = add_colab_links.colab_cell(
        add_colab_links.REPO_ROOT / "OCR/old_name.ipynb"
    )
    notebook = {"cells": [stale, {"cell_type": "markdown", "source": ["# Title\n"]}]}

    assert add_colab_links.sync_colab_link(notebook, path)
    assert notebook["cells"][0] == add_colab_links.colab_cell(path)

    unchanged = copy.deepcopy(notebook)
    assert not add_colab_links.sync_colab_link(unchanged, path)
    assert unchanged == notebook

    mixed = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "keep-me"},
                "source": [
                    "# Title\n",
                    "\n",
                    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                    "(https://colab.research.google.com/github/Gan4x4/cv/blob/main/old_name.ipynb)",
                ],
            }
        ]
    }
    assert add_colab_links.sync_colab_link(mixed, path)
    assert mixed["cells"][0]["source"][0] == "# Title\n"
    assert mixed["cells"][0]["metadata"] == {"id": "keep-me"}
    assert add_colab_links.colab_url(path) in mixed["cells"][0]["source"][2]
    mixed_after_update = copy.deepcopy(mixed)
    assert not add_colab_links.sync_colab_link(mixed_after_update, path)
    assert mixed_after_update == mixed

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
