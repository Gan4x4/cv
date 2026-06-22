#!/usr/bin/env python3
"""Convert a Jupyter notebook to Markdown.

The script preserves markdown cells, code cells, and image outputs. It ignores
notebook metadata, execution counts, and cell tags.



Usage:
    python scripts/convert2md.py
    python scripts/convert2md.py --root path/to/repository --output-dir path/to/md


Image path replacement: after generation, output data are uploaded to the server;
use SERVER_URL for image URLs instead of local paths.

"""




from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

SERVER_URL = "https://ml.gan4x4.ru/wb/cv/md/"


def read_notebook(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_source(source: Any) -> str:
    if isinstance(source, list):
        return "".join(source)
    return source or ""


HTML_IMAGE_TAG_RE = re.compile(
    r'<img\b[^>]*\bsrc\s*=\s*["\']?([^"\'\s>]+)["\']?[^>]*>',
    re.IGNORECASE,
)

COLAB_BADGE_HTML_RE = re.compile(
    r"""^\s*<a\b[^>]*\bhref\s*=\s*["']([^"']+)["'][^>]*>\s*
        <img\b[^>]*\bsrc\s*=\s*["']([^"']*colab-badge\.svg[^"']*)["'][^>]*>\s*</a>""",
    re.IGNORECASE | re.VERBOSE,
)
SCRIPT_TAG_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
CENTER_TAG_RE = re.compile(r"</?center\b[^>]*>", re.IGNORECASE)
COLAB_DATAFRAME_RE = re.compile(r"<div\b[^>]*\bcolab-df-container\b", re.IGNORECASE)


def convert_html_images_to_markdown(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        src = match.group(1)
        alt_match = re.search(
            r'\balt\s*=\s*["\']?([^"\'>]+)["\']?',
            match.group(0),
            re.IGNORECASE,
        )
        alt = alt_match.group(1).strip() if alt_match else "image"
        return f"![{alt}]({src})"

    return HTML_IMAGE_TAG_RE.sub(replace, text)


def convert_leading_colab_badge(text: str) -> str:
    """Convert an optional HTML Colab launch badge in the first cell to a text link."""
    return COLAB_BADGE_HTML_RE.sub(
        lambda match: f"[Open In Colab]({match.group(1)})",
        text,
        count=1,
    )


def remove_unsupported_html(text: str) -> str:
    """Drop JavaScript and alignment wrappers that Markdown renderers do not support."""
    return CENTER_TAG_RE.sub("", SCRIPT_TAG_RE.sub("", text))


def render_text_output(text: Any) -> str:
    if isinstance(text, list):
        text = "".join(text)
    return str(text)


def save_image(
    output_key: str,
    data: bytes,
    output_dir: Path,
    notebook_name: str,
    cell_index: int,
    output_index: int,
    output_root: Path | None = None,
) -> str:
    ext = output_key.split("/")[-1]
    if ext == "svg+xml":
        ext = "svg"
    elif ext == "jpeg":
        ext = "jpg"

    image_dir = output_dir / "outputs"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_name = f"{notebook_name}_cell_{cell_index + 1}_output_{output_index + 1}.{ext}"
    image_path = image_dir / image_name
    image_path.write_bytes(data)

    relative_dir = output_dir.relative_to(output_root) if output_root else Path()
    url_dir = "" if relative_dir == Path() else f"{relative_dir.as_posix()}/"
    return f"{SERVER_URL}{url_dir}outputs/{image_name}"


def render_output_data(
    output_data: dict[str, Any],
    output_dir: Path,
    notebook_name: str,
    cell_index: int,
    output_index: int,
    output_root: Path | None = None,
) -> list[str]:
    parts: list[str] = []

    for mime_type, value in output_data.items():
        if mime_type == "text/plain":
            text = render_text_output(value)
            if text.strip():
                parts.append(f"```text\n{text.rstrip()}\n```")
        elif mime_type == "text/html":
            if COLAB_DATAFRAME_RE.search(render_text_output(value)):
                continue
            html = convert_html_images_to_markdown(
                remove_unsupported_html(render_text_output(value).rstrip())
            )
            if html:
                parts.append(html)
        elif mime_type == "text/markdown":
            md = remove_unsupported_html(render_text_output(value).rstrip())
            if md:
                parts.append(md)
        elif mime_type.startswith("image/") and isinstance(value, str):
            try:
                image_bytes = base64.b64decode(value)
            except (ValueError, TypeError):
                continue
            relative_path = save_image(
                mime_type,
                image_bytes,
                output_dir,
                notebook_name,
                cell_index,
                output_index,
                output_root,
            )
            parts.append(f"![image]({relative_path})")

    return parts


def render_output(
    output: dict[str, Any],
    output_dir: Path,
    notebook_name: str,
    cell_index: int,
    output_index: int,
    output_root: Path | None = None,
) -> list[str]:
    output_type = output.get("output_type")

    if output_type == "stream":
        text = render_text_output(output.get("text", ""))
        if not text:
            return []
        stream_name = output.get("name", "stdout")
        return [f"```{stream_name}\n{text.rstrip()}\n```"]

    if output_type in {"execute_result", "display_data"}:
        return render_output_data(
            output.get("data", {}),
            output_dir,
            notebook_name,
            cell_index,
            output_index,
            output_root,
        )

    if output_type == "error":
        traceback = output.get("traceback", [])
        if traceback:
            return [f"```text\n{chr(10).join(traceback)}\n```"]

    return []


def convert_notebook_to_markdown(
    notebook_path: Path, output_path: Path | None = None, output_root: Path | None = None
) -> Path:
    notebook = read_notebook(notebook_path)
    notebook_name = notebook_path.stem

    if output_path is None:
        output_path = notebook_path.with_suffix(".md")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_path.parent

    markdown_lines: list[str] = []

    for cell_index, cell in enumerate(notebook.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = normalize_source(cell.get("source", ""))

        if cell_type == "markdown":
            if cell_index == 0:
                source = convert_leading_colab_badge(source)
            if source.strip():
                markdown_lines.append(
                    convert_html_images_to_markdown(remove_unsupported_html(source.rstrip()))
                )
                markdown_lines.append("")
            continue

        if cell_type == "code":
            markdown_lines.append(f"```python")
            markdown_lines.append(source.rstrip())
            markdown_lines.append("```")
            markdown_lines.append("")

            outputs = cell.get("outputs", [])
            for output_index, output in enumerate(outputs):
                rendered = render_output(
                    output, output_dir, notebook_name, cell_index, output_index, output_root
                )
                for block in rendered:
                    markdown_lines.append(block)
                    markdown_lines.append("")

    content = "\n".join(markdown_lines).strip() + "\n"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert every Jupyter notebook in a repository into Markdown."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository to scan (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Markdown directory (default: <root>/md).",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Error: repository directory not found: {root}", file=sys.stderr)
        return 1

    output_dir = (args.output_dir or root / "md").resolve()
    notebooks = sorted(
        path for path in root.rglob("*.ipynb") if output_dir not in path.parents
    )
    if not notebooks:
        print(f"No notebooks found in {root}")
        return 0

    for notebook_path in notebooks:
        output_path = output_dir / notebook_path.relative_to(root).with_suffix(".md")
        convert_notebook_to_markdown(notebook_path, output_path, output_dir)
        print(f"Converted {notebook_path.relative_to(root)} -> {output_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
