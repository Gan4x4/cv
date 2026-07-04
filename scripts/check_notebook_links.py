#!/usr/bin/env python3
"""Find broken URLs embedded in Jupyter notebooks.

The script scans every `.ipynb` file under the repository root, extracts URL
strings from markdown cells, and probes each URL with a lightweight request. It
prints URLs whose final server response is not HTTP 200.

Usage:
    python scripts/check_notebook_links.py [--root PATH] [--url-filter TEXT]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import orjson
except ImportError:  # pragma: no cover - optional dependency
    orjson = None

try:
    from urlextract import URLExtract
except ImportError:  # pragma: no cover - optional dependency
    URLExtract = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_kwargs):
        return iterable


URL_RE = re.compile(r"https?://[^\s<>'\"\)\]]+")
DEFAULT_ROOT = Path(__file__).resolve().parent.parent
URL_EXTRACTOR = URLExtract() if URLExtract is not None else None
COLAB_REPO_PATH = "/github/Gan4x4/cv/blob/main/"


def progress(iterable, **kwargs):
    return tqdm(
        iterable,
        disable=False,
        file=sys.stderr,
        dynamic_ncols=True,
        **kwargs,
    )


@dataclass(frozen=True)
class ProbeResult:
    url: str
    status: int | None


@dataclass(frozen=True)
class UrlLocation:
    notebook: str
    cell: int
    topic: str | None = None


@dataclass(frozen=True)
class UrlOccurrence:
    url: str
    location: UrlLocation


def normalize_url(url: str) -> str:
    url = url.strip()
    while url and url[-1] in ".,;:!?)]'\"":
        url = url[:-1]
    return url


def extract_urls_from_text_simple(text: str) -> set[str]:
    if URL_EXTRACTOR is not None:
        return {normalize_url(url) for url in URL_EXTRACTOR.find_urls(text)}
    return {normalize_url(match.group(0)) for match in URL_RE.finditer(text)}


def heading_from_text(text: str) -> str | None:
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if match:
            return match.group(1)
    return None


def format_location(location: UrlLocation) -> str:
    parts = [location.notebook, f"cell {location.cell}"]
    if location.topic:
        parts.append(f'topic "{location.topic}"')
    return ", ".join(parts)


def format_locations(locations: set[UrlLocation]) -> str:
    return "; ".join(
        format_location(location)
        for location in sorted(locations, key=lambda loc: (loc.notebook, loc.cell))
    )


def extract_urls_from_notebook(path: Path, root: Path) -> list[UrlOccurrence]:
    if orjson is None:
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = orjson.loads(path.read_bytes())

    occurrences: set[UrlOccurrence] = set()
    relative_path = str(path.relative_to(root))
    current_topic: str | None = None
    for cell_number, cell in enumerate(data.get("cells", []), start=1):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        text = "".join(str(part) for part in source) if isinstance(source, list) else str(source or "")
        current_topic = heading_from_text(text) or current_topic
        location = UrlLocation(relative_path, cell_number, current_topic)
        if isinstance(source, list):
            for line in source:
                if "http" in line:
                    for url in extract_urls_from_text_simple(line):
                        occurrences.add(UrlOccurrence(url, location))
        elif isinstance(source, str) and "http" in source:
            for url in extract_urls_from_text_simple(source):
                occurrences.add(UrlOccurrence(url, location))
    return sorted(occurrences, key=lambda item: (item.url, item.location.cell))


def url_matches_filter(url: str, url_filter: str) -> bool:
    if not url_filter:
        return True
    return url_filter in url


def colab_relative_path(url: str) -> Path | None:
    parsed = urlsplit(url)
    if parsed.netloc != "colab.research.google.com":
        return None
    if not parsed.path.startswith(COLAB_REPO_PATH):
        return None
    return Path(unquote(parsed.path.removeprefix(COLAB_REPO_PATH)))


def probe_once(url: str, method: str, timeout: float) -> int | None:
    request = Request(url, method=method)
    request.add_header("User-Agent", "Mozilla/5.0 (link-check)")
    if method == "GET":
        request.add_header("Range", "bytes=0-0")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status
    except HTTPError as exc:
        return exc.code
    except (URLError, TimeoutError, ValueError, OSError):
        return None


def probe_url(url: str, timeout: float) -> ProbeResult:
    status = probe_once(url, "HEAD", timeout)
    if status is None or status in {400, 403, 405, 501}:
        fallback = probe_once(url, "GET", timeout)
        if fallback is not None:
            status = fallback
    return ProbeResult(url=url, status=status)


def probe_urls(urls: list[str], timeout: float, max_workers: int) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(probe_url, url, timeout): url
            for url in urls
        }
        for future in progress(
            concurrent.futures.as_completed(future_to_url),
            total=len(future_to_url),
            desc="Probe URLs",
            unit="url",
        ):
            try:
                results.append(future.result())
            except Exception:
                results.append(ProbeResult(url=future_to_url[future], status=None))
    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Extract and probe URLs embedded in notebook files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Repository root to scan (defaults to the repo root next to this script)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print notebook and URL extraction progress to stderr",
        default = True
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Maximum number of concurrent URL probes",
    )
    parser.add_argument(
        "--url-filter",
        default="ml.gan4x4.ru",
        help="Check only URLs containing this substring; empty string checks all URLs",
    )
    args = parser.parse_args(argv)

    notebook_paths = sorted(
        p for p in args.root.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts
    )

    url_to_locations: dict[str, set[UrlLocation]] = {}
    broken_colab_links: list[UrlOccurrence] = []
    extractor = extract_urls_from_notebook

    for path in progress(notebook_paths, desc="Notebooks", unit="file"):
        try:
            occurrences = extractor(path, args.root)
        except Exception as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue
        relative_path = str(path.relative_to(args.root))
        if args.verbose:
            print(f"[scan] {relative_path}: {len(occurrences)} urls", file=sys.stderr)
        for occurrence in occurrences:
            url = occurrence.url
            colab_path = colab_relative_path(url)
            if colab_path is not None and not (args.root / colab_path).is_file():
                broken_colab_links.append(occurrence)
            if not url_matches_filter(url, args.url_filter):
                continue
            url_to_locations.setdefault(url, set()).add(occurrence.location)

    if args.verbose:
        print(
            f"[summary] notebooks={len(notebook_paths)} urls={len(url_to_locations)} filter={args.url_filter!r}",
            file=sys.stderr,
        )

    urls_to_probe = sorted(url_to_locations)
    if args.verbose:
        print(f"[summary] probing {len(urls_to_probe)} unique urls with {args.workers} workers", file=sys.stderr)

    results = probe_urls(urls_to_probe, args.timeout, args.workers)

    broken = 0
    for occurrence in broken_colab_links:
        broken += 1
        print(f"MISSING_COLAB {occurrence.url} [{format_location(occurrence.location)}]")

    for result in results:
        if result.status != 200:
            broken += 1
            locations = format_locations(url_to_locations.get(result.url, set()))
            status = result.status if result.status is not None else "ERR"
            print(f"{status} {result.url} [{locations}]")

    print(f"[summary] broken links found: {broken}", file=sys.stderr)
    if args.verbose and broken == 0:
        print("[summary] no broken links found", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
