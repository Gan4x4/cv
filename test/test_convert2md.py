import base64
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from scripts import convert2md


class Convert2MdTests(unittest.TestCase):
    def test_normalize_source_accepts_notebook_lines_and_strings(self):
        self.assertEqual(convert2md.normalize_source(["first\n", "second"]), "first\nsecond")
        self.assertEqual(convert2md.normalize_source("already joined"), "already joined")
        self.assertEqual(convert2md.normalize_source(None), "")

    def test_html_cleanup_and_image_conversion(self):
        source = (
            '<center><img src="diagram.png" alt="Model diagram"></center>'
            "<script>ignored()</script>"
        )

        cleaned = convert2md.remove_unsupported_html(source)

        self.assertEqual(
            convert2md.convert_html_images_to_markdown(cleaned),
            "![Model diagram](diagram.png)",
        )

    def test_leading_colab_badge_is_converted_to_link(self):
        source = (
            '<a href="https://colab.research.google.com/example">'
            '<img src="https://colab.research.google.com/assets/colab-badge.svg">'
            "</a>\n\n# Notebook"
        )

        result = convert2md.convert_leading_colab_badge(source)

        self.assertEqual(
            result,
            "[Open In Colab](https://colab.research.google.com/example)\n\n# Notebook",
        )

    def test_matplotlib_figure_repr_is_not_rendered_as_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            figure_parts = convert2md.render_output_data(
                {"text/plain": "<Figure size 640x480 with 1 Axes>"},
                output_dir,
                "example",
                0,
                0,
            )
            regular_text_parts = convert2md.render_output_data(
                {"text/plain": "useful result"},
                output_dir,
                "example",
                0,
                0,
            )

        self.assertEqual(figure_parts, [])
        self.assertEqual(regular_text_parts, ["```text\nuseful result\n```"])

    def test_risky_list_continuation_warns_without_changing_output(self):
        source = (
            "- One stage: anchors,\n"
            "\n"
            "    image + anchors produces bounding boxes.\n"
            "- Anchor free: bounding boxes."
        )
        notebook = {
            "cells": [{"cell_type": "markdown", "source": source}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            notebook_path = root / "example.ipynb"
            output_path = root / "example.md"
            notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
            stderr = StringIO()

            with redirect_stderr(stderr):
                convert2md.convert_notebook_to_markdown(notebook_path, output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), source + "\n")
            self.assertIn(
                f"Warning: {notebook_path}: cell 1, line 1: "
                'near "One stage: anchors,": risky list break',
                stderr.getvalue(),
            )

    def test_risky_list_continuation_check_avoids_false_positives(self):
        source = (
            "- Normal item\n"
            "    - Nested item\n"
            "\n"
            "Unindented paragraph\n"
            "\n"
            "```markdown\n"
            "- Example item\n"
            "\n"
            "    Example continuation\n"
            "```\n"
            "- Item before fenced code\n"
            "\n"
            "    ```python\n"
            "    print('example')\n"
            "    ```"
        )

        self.assertEqual(convert2md.find_risky_list_continuations(source), [])

    def test_convert_notebook_renders_cells_outputs_and_image(self):
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Example\n", '<img src="source.png" alt="Source">'],
                },
                {
                    "cell_type": "code",
                    "source": ["print('hello')"],
                    "outputs": [
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": ["hello\n"],
                        },
                        {
                            "output_type": "display_data",
                            "data": {
                                "image/png": base64.b64encode(b"png data").decode("ascii")
                            },
                        },
                        {
                            "output_type": "error",
                            "traceback": ["ValueError: bad value"],
                        },
                    ],
                },
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            notebook_path = root / "example.ipynb"
            output_root = root / "md"
            output_path = output_root / "example.md"
            notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

            result = convert2md.convert_notebook_to_markdown(
                notebook_path, output_path, output_root
            )

            self.assertEqual(result, output_path)
            markdown = output_path.read_text(encoding="utf-8")
            self.assertIn("# Example", markdown)
            self.assertIn("![Source](source.png)", markdown)
            self.assertIn("```python\nprint('hello')\n```", markdown)
            self.assertIn("```stdout\nhello\n```", markdown)
            self.assertIn("```text\nValueError: bad value\n```", markdown)
            self.assertIn(
                f"![image]({convert2md.SERVER_URL}outputs/"
                "example_cell_2_output_2.png)",
                markdown,
            )
            self.assertEqual(
                (output_root / "outputs" / "example_cell_2_output_2.png").read_bytes(),
                b"png data",
            )

    def test_main_preserves_notebook_subdirectories(self):
        notebook = {
            "cells": [{"cell_type": "markdown", "source": "# Nested"}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            notebook_path = root / "topic" / "lesson.ipynb"
            notebook_path.parent.mkdir()
            notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = convert2md.main(["--root", str(root)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                (root / "md" / "topic" / "lesson.md").read_text(encoding="utf-8"),
                "# Nested\n",
            )
            self.assertEqual(stdout.getvalue(), "Converted 1 file.\n")

    def test_main_prints_zero_converted_files_for_empty_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = convert2md.main(["--root", temp_dir])

            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout.getvalue(), "Converted 0 files.\n")


if __name__ == "__main__":
    unittest.main()
