#!/usr/bin/env python3
"""
WB Tech materials scraper with topic folders.

Default: scrapes only first 5 visible menu items.
To scrape all: set MAX_ITEMS = None.

Output example:
  wb_materials_md/
    001_topic/
      _index.md          # if topic itself has content
      002_lesson.md
      003_subtopic/
        004_lesson.md
"""

from __future__ import annotations

import re
import sys
import hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


URL = "https://ts.wb.ru/courses/3/materials"

OUT_DIR = Path("wb_materials_md")
PROFILE_DIR = Path(".pw-wb-profile")

MAX_ITEMS = None  # set None to scrape all items

TREE_ITEM = ".wb-materials-tree-item"
TREE_TITLE = ".wb-materials-tree-item__title"
EDITOR_TITLE = ".wb-materials-editor-form__title textarea"
EDITOR_TYPE = ".wb-materials-editor-tag__name"

CONTENT_SELECTOR = (
    ".wb-materials-editor-form__content .bn-editor, "
    ".wb-materials-editor-form__content .ProseMirror, "
    ".wb-materials-editor-form__content"
)


EXTRACT_MD_JS = r"""
() => {
  const root =
    document.querySelector('.wb-materials-editor-form__content .bn-editor') ||
    document.querySelector('.wb-materials-editor-form__content .ProseMirror') ||
    document.querySelector('.wb-materials-editor-form__content');

  if (!root) return '';

  function clean(s) {
    return (s || '')
      .replace(/\u00a0/g, ' ')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }

  function esc(s) {
    return (s || '').replace(/\[/g, '\\[').replace(/\]/g, '\\]');
  }

  function inline(node) {
    if (!node) return '';
    if (node.nodeType === Node.TEXT_NODE) return node.nodeValue || '';
    if (node.nodeType !== Node.ELEMENT_NODE) return '';

    const tag = node.tagName.toLowerCase();

    if (tag === 'br') return '\n';

    if (tag === 'a') {
      const href = node.getAttribute('href') || '';
      const txt = clean(Array.from(node.childNodes).map(inline).join('')) || href;
      return href ? `[${esc(txt)}](${href})` : txt;
    }

    if (tag === 'strong' || tag === 'b') {
      return `**${Array.from(node.childNodes).map(inline).join('')}**`;
    }

    if (tag === 'em' || tag === 'i') {
      return `*${Array.from(node.childNodes).map(inline).join('')}*`;
    }

    if (tag === 'code') {
      return '`' + (node.textContent || '').replace(/`/g, '\\`') + '`';
    }

    if (tag === 'img') {
      const src = node.getAttribute('src') || '';
      const alt = node.getAttribute('alt') || '';
      return src ? `![${esc(alt)}](${src})` : '';
    }

    return Array.from(node.childNodes).map(inline).join('');
  }

  function tableToMd(table) {
    const rows = Array.from(table.querySelectorAll('tr')).map(tr =>
      Array.from(tr.querySelectorAll('th,td')).map(td => clean(td.textContent).replace(/\|/g, '\\|'))
    ).filter(r => r.length);

    if (!rows.length) return '';

    const width = Math.max(...rows.map(r => r.length));
    const norm = rows.map(r => [...r, ...Array(width - r.length).fill('')]);
    const header = norm[0];
    const sep = Array(width).fill('---');

    return [
      '| ' + header.join(' | ') + ' |',
      '| ' + sep.join(' | ') + ' |',
      ...norm.slice(1).map(r => '| ' + r.join(' | ') + ' |')
    ].join('\n');
  }

  function blockToMd(block) {
    const type = (block.getAttribute('data-content-type') || '').toLowerCase();
    const cls = (block.className || '').toString().toLowerCase();

    const table = block.querySelector('table');
    if (table) return tableToMd(table);

    const pre = block.querySelector('pre');
    if (pre) return '```\n' + clean(pre.textContent || '') + '\n```';

    if (type.includes('code') || cls.includes('code')) {
      const txt = clean(block.textContent || '');
      return txt ? '```\n' + txt + '\n```' : '';
    }

    const h = block.querySelector('h1,h2,h3,h4,h5,h6');
    if (h) {
      const level = Number(h.tagName.substring(1)) || 2;
      return '#'.repeat(level) + ' ' + clean(Array.from(h.childNodes).map(inline).join(''));
    }

    let text = clean(Array.from(block.childNodes).map(inline).join(''));
    if (!text) text = clean(block.innerText || block.textContent || '');
    if (!text) return '';

    if (type.includes('bullet') || cls.includes('bullet')) return '- ' + text;
    if (type.includes('number') || cls.includes('number')) return '1. ' + text;
    if (type.includes('check') || cls.includes('check')) return '- [ ] ' + text;

    return text;
  }

  let blocks = Array.from(root.querySelectorAll('.bn-block-content'));
  if (!blocks.length) {
    blocks = Array.from(root.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, pre, table, .ce-block, .bn-block'));
  }
  if (!blocks.length) {
    return clean(root.innerText || root.textContent || '');
  }

  const parts = blocks.map(blockToMd).map(clean).filter(Boolean);
  return clean(parts.join('\n\n')) || clean(root.innerText || root.textContent || '');
}
"""


def slugify(text: str, max_len: int = 90) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\\/:*?\"<>|]+", " ", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._ ")
    return (text or "untitled")[:max_len].strip("_")


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:8]


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    for i in range(2, 1000):
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate

    return parent / f"{stem}_{short_hash(str(path))}{suffix}"


def safe_input_or_text(page, selector: str, default: str = "") -> str:
    loc = page.locator(selector)
    try:
        if loc.count() == 0:
            return default
        return loc.first.input_value(timeout=1000).strip()
    except Exception:
        try:
            return loc.first.inner_text(timeout=1000).strip()
        except Exception:
            return default


def wait_for_login_if_needed(page) -> None:
    try:
        page.wait_for_selector(TREE_ITEM, timeout=12000)
        return
    except PlaywrightTimeoutError:
        print()
        print("Login/phone-code confirmation is probably required.")
        print("Complete it in the opened browser, then press Enter here.")
        input()
        page.wait_for_selector(TREE_ITEM, timeout=120000)


def expand_all_folders(page, max_rounds: int = 30) -> None:
    for _ in range(max_rounds):
        before = page.locator(TREE_ITEM).count()

        clicked = page.evaluate(
            """
            () => {
              function visible(el) {
                return !!el && !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
              }

              let clicked = 0;
              const items = Array.from(document.querySelectorAll('.wb-materials-tree-item[data-has-children="true"]'));

              for (const item of items) {
                if (!visible(item)) continue;

                const next = item.nextElementSibling;
                const alreadyOpen =
                  next &&
                  next.classList.contains('wb-materials-tree') &&
                  visible(next) &&
                  next.querySelector('.wb-materials-tree-item');

                if (alreadyOpen) continue;

                const toggle =
                  item.querySelector('.wb-materials-tree-item__toggle') ||
                  item.querySelector('.wb-materials-tree-item__icons') ||
                  item;

                toggle.dispatchEvent(new MouseEvent('click', {
                  bubbles: true,
                  cancelable: true,
                  view: window
                }));
                clicked += 1;
              }

              return clicked;
            }
            """
        )

        page.wait_for_timeout(500)
        after = page.locator(TREE_ITEM).count()

        if clicked == 0 and after == before:
            break


def collect_items(page) -> list[dict]:
    return page.evaluate(
        """
        () => Array.from(document.querySelectorAll('.wb-materials-tree-item')).map((el, idx) => {
          const title = el.querySelector('.wb-materials-tree-item__title')?.innerText?.trim() || '';
          const rect = el.getBoundingClientRect();

          let levelRaw = el.getAttribute('data-level');
          let level = Number(levelRaw);

          if (!Number.isFinite(level)) {
            // Fallback: estimate by nesting in tree containers.
            level = 0;
            let p = el.parentElement;
            while (p) {
              if (p.classList && p.classList.contains('wb-materials-tree')) level += 1;
              p = p.parentElement;
            }
            level = Math.max(0, level - 1);
          }

          return {
            idx,
            title,
            type: el.getAttribute('data-type') || '',
            level,
            hasChildren: el.getAttribute('data-has-children') === 'true',
            visible: !!(rect.width || rect.height)
          };
        }).filter(x => x.title && x.visible)
        """
    )


def click_menu_item(page, idx: int) -> None:
    item = page.locator(TREE_ITEM).nth(idx)
    item.scroll_into_view_if_needed(timeout=5000)

    title = item.locator(TREE_TITLE)
    if title.count() > 0:
        title.first.click(timeout=5000)
    else:
        item.click(timeout=5000)

    page.wait_for_timeout(1200)


def save_debug_html(page, order: int, title: str) -> None:
    html = page.evaluate(
        """
        () => {
          const el =
            document.querySelector('.wb-materials-editor-form__content') ||
            document.querySelector('.wb-content-wrapper') ||
            document.body;
          return el ? el.outerHTML : document.documentElement.outerHTML;
        }
        """
    )
    debug_name = f"_debug_empty_{order:03d}_{slugify(title, 50)}.html"
    (OUT_DIR / debug_name).write_text(html, encoding="utf-8")


def build_output_plan(items: list[dict]) -> list[dict]:
    """
    Add output path info using menu hierarchy:
    - items with children -> folder path
    - leaf items -> markdown file inside nearest parent folder
    """
    folder_stack: list[Path] = []
    plan: list[dict] = []

    for order, info in enumerate(items, start=1):
        level = int(info.get("level", 0))
        title = info["title"]
        is_folder = bool(info.get("hasChildren"))

        while len(folder_stack) > level:
            folder_stack.pop()

        folder_name = f"{order:03d}_{slugify(title, 70)}"

        if is_folder:
            folder_path = OUT_DIR.joinpath(*folder_stack, folder_name)
            info["output_dir"] = folder_path
            info["output_file"] = folder_path / "_index.md"
            info["is_folder"] = True

            folder_stack = folder_stack[:level]
            folder_stack.append(Path(folder_name))
        else:
            parent_dir = OUT_DIR.joinpath(*folder_stack)
            file_name = f"{order:03d}_{slugify(title, 80)}.md"
            info["output_dir"] = parent_dir
            info["output_file"] = parent_dir / file_name
            info["is_folder"] = False

        plan.append(info)

    return plan


def write_global_index(saved_files: list[Path]) -> None:
    lines = ["# Scraped WB materials", ""]
    for path in sorted(saved_files):
        rel = path.relative_to(OUT_DIR)
        lines.append(f"- [{rel}]({rel.as_posix()})")
    (OUT_DIR / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            viewport={"width": 1600, "height": 1000},
            accept_downloads=False,
        )

        page = context.pages[0] if context.pages else context.new_page()

        print(f"Opening: {URL}")
        page.goto(URL, wait_until="domcontentloaded")

        wait_for_login_if_needed(page)

        print("Expanding nested folders...")
        expand_all_folders(page)

        items = collect_items(page)
        print(f"Found visible menu items: {len(items)}")

        items_to_scrape = items[:MAX_ITEMS] if MAX_ITEMS else items
        plan = build_output_plan(items_to_scrape)

        print(f"Will scrape: {len(plan)}")
        print(f"Output root: {OUT_DIR.resolve()}")

        saved = 0
        skipped = 0
        failed = []
        saved_files: list[Path] = []

        for order, info in enumerate(plan, start=1):
            idx = int(info["idx"])
            menu_title = info["title"]
            level = int(info.get("level", 0))
            is_folder = bool(info.get("is_folder"))

            kind = "folder" if is_folder else "file"
            print(f"[{order:03d}/{len(plan):03d}] level={level} {kind}: {menu_title}")

            try:
                click_menu_item(page, idx)

                editor_title = safe_input_or_text(page, EDITOR_TITLE, menu_title) or menu_title
                editor_type = safe_input_or_text(page, EDITOR_TYPE, info.get("type", ""))

                body = page.evaluate(EXTRACT_MD_JS).strip()
                print(f"  content chars: {len(body)}")

                if not body:
                    skipped += 1
                    save_debug_html(page, order, menu_title)
                    print("  - empty; debug html saved")
                    continue

                output_dir = Path(info["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)

                output_file = Path(info["output_file"])
                output_file = unique_path(output_file)

                md = f"# {editor_title}\n\n{body}\n\n"
                md += f"<!-- source: {URL}; menu: {menu_title}; type: {editor_type}; level: {level}; folder: {is_folder} -->\n"

                output_file.write_text(md, encoding="utf-8")
                saved_files.append(output_file)
                saved += 1

                print(f"  + saved: {output_file.relative_to(OUT_DIR)}")

            except Exception as e:
                failed.append(f"{menu_title}: {repr(e)}")
                print(f"  ! failed: {repr(e)}")

        write_global_index(saved_files)

        if failed:
            (OUT_DIR / "_failed.txt").write_text("\n".join(failed), encoding="utf-8")

        print()
        print(f"Saved: {saved}")
        print(f"Skipped empty: {skipped}")
        print(f"Failed: {len(failed)}")
        print(f"Output folder: {OUT_DIR.resolve()}")

        context.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped.")
