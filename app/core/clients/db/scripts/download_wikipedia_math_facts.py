"""
Скачивает математические факты из Wikipedia через MediaWiki API
и сохраняет в формат knowledge_base.

Запуск:
  python app/core/clients/db/scripts/download_wikipedia_math_facts.py
  python app/core/clients/db/scripts/download_wikipedia_math_facts.py --target 1100
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path
from typing import Any

API_URL = "https://en.wikipedia.org/w/api.php"

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
OUTPUT_FILE = KNOWLEDGE_BASE_DIR / "wikipedia_math_facts.json"

# Даем запас по кандидатам, чтобы после фильтрации получить >= target
OVERFETCH_FACTOR = 2.0
MAX_EXTRACT_LEN = 900

CATEGORY_SOURCES = [
    "Category:Mathematical_theorems",
    "Category:Geometry_theorems",
    "Category:Algebra_theorems",
    "Category:Number_theory",
    "Category:Mathematical_analysis",
    "Category:Linear_algebra",
    "Category:Combinatorics",
    "Category:Probability_theory",
    "Category:Statistics_theory",
    "Category:Topology",
    "Category:Graph_theory",
    "Category:Set_theory",
    "Category:Mathematical_logic",
    "Category:Inequalities",
    "Category:Trigonometry",
]

THEOREM_HINTS = (
    "theorem",
    "lemma",
    "corollary",
    "law",
    "inequality",
    "identity",
    "formula",
    "rule",
    "criterion",
    "conjecture",
    "postulate",
    "principle",
)
AXIOM_HINTS = ("axiom", "postulate")
DEFINITION_HINTS = (
    "definition",
    "concept",
    "space",
    "function",
    "set",
    "sequence",
    "matrix",
    "algebra",
    "topology",
)


def _request_json(params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={
            "User-Agent": "ProofsAnalyzeProj-RAG-Bot/1.0 (academic project)",
            "Accept": "application/json",
        },
    )
    backoff = 1.0
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 20.0)
                continue
            raise
        except URLError:
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 20.0)
                continue
            raise
    return {}


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= MAX_EXTRACT_LEN:
        return text
    cutoff = text.rfind(".", 0, MAX_EXTRACT_LEN)
    if cutoff == -1:
        return text[:MAX_EXTRACT_LEN].rstrip() + "..."
    return text[: cutoff + 1].strip()


def _infer_type(title: str, categories: list[str]) -> str:
    lower_title = title.lower()
    categories_joined = " ".join(c.lower() for c in categories)
    joined = f"{lower_title} {categories_joined}"
    if any(h in joined for h in AXIOM_HINTS):
        return "axiom"
    if any(h in joined for h in THEOREM_HINTS):
        return "theorem"
    if any(h in joined for h in DEFINITION_HINTS):
        return "definition"
    return "definition"


def _collect_titles_from_category(category: str, max_titles: int) -> list[str]:
    titles: list[str] = []
    cont: dict[str, str] | None = None
    while len(titles) < max_titles:
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "page",
            "cmlimit": "500",
        }
        if cont:
            params.update(cont)
        payload = _request_json(params)
        members = payload.get("query", {}).get("categorymembers", [])
        if not members:
            break
        titles.extend(str(item.get("title", "")).strip() for item in members)
        titles = [t for t in titles if t]
        if "continue" not in payload:
            break
        cont = {
            k: v
            for k, v in payload["continue"].items()
            if k in ("cmcontinue", "continue")
        }
    return titles[:max_titles]


def _fetch_page_info(title: str) -> dict[str, Any] | None:
    params: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "prop": "extracts|categories",
        "explaintext": "1",
        "exintro": "1",
        "titles": title,
        "cllimit": "200",
    }
    try:
        payload = _request_json(params)
    except Exception:
        return None
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    if "missing" in page:
        return None
    extract = _clean_text(str(page.get("extract", "")))
    if not extract or len(extract) < 80:
        return None
    categories = [
        c.get("title", "").replace("Category:", "").strip()
        for c in page.get("categories", [])
        if isinstance(c, dict) and c.get("title")
    ]
    return {
        "title": str(page.get("title", title)).strip(),
        "extract": extract,
        "categories": categories,
    }


def _fetch_pages_info_batch(titles: list[str]) -> list[dict[str, Any]]:
    if not titles:
        return []
    params: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "prop": "extracts|categories",
        "explaintext": "1",
        "exintro": "1",
        "titles": "|".join(titles),
        "cllimit": "200",
    }
    try:
        payload = _request_json(params)
    except Exception:
        return []
    pages = payload.get("query", {}).get("pages", {})
    results: list[dict[str, Any]] = []
    for page in pages.values():
        if not isinstance(page, dict) or "missing" in page:
            continue
        extract = _clean_text(str(page.get("extract", "")))
        if not extract or len(extract) < 80:
            continue
        categories = [
            c.get("title", "").replace("Category:", "").strip()
            for c in page.get("categories", [])
            if isinstance(c, dict) and c.get("title")
        ]
        results.append(
            {
                "title": str(page.get("title", "")).strip(),
                "extract": extract,
                "categories": categories,
            }
        )
    return results


def build_wikipedia_facts(
    target: int = 1000,
    delay_s: float = 0.03,
    progress_file: Path | None = None,
    batch_size: int = 20,
) -> list[dict[str, str]]:
    wanted_candidates = max(int(target * OVERFETCH_FACTOR), target + 300)
    per_category = max(150, wanted_candidates // max(1, len(CATEGORY_SOURCES)))
    all_titles: list[str] = []
    print(f"Collecting titles from {len(CATEGORY_SOURCES)} categories...")
    for category in CATEGORY_SOURCES:
        titles = _collect_titles_from_category(category, max_titles=per_category)
        print(f"  {category}: {len(titles)} titles")
        all_titles.extend(titles)
    # Дедуп по порядку
    seen_titles: set[str] = set()
    deduped_titles: list[str] = []
    for title in all_titles:
        key = title.casefold()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        deduped_titles.append(title)
    print(f"Unique titles: {len(deduped_titles)}")

    items: list[dict[str, str]] = []
    used_names: set[str] = set()
    total_titles = len(deduped_titles)
    for start in range(0, total_titles, max(1, batch_size)):
        if len(items) >= target:
            break
        chunk = deduped_titles[start : start + max(1, batch_size)]
        infos = _fetch_pages_info_batch(chunk)
        for info in infos:
            if len(items) >= target:
                break
            name = info["title"].strip()
            if not name:
                continue
            name_key = name.casefold()
            if name_key in used_names:
                continue
            used_names.add(name_key)
            item_type = _infer_type(name, info["categories"])
            category_hint = next((c for c in info["categories"] if c), "Wikipedia")
            items.append(
                {
                    "type": item_type,
                    "name": name,
                    "statement": info["extract"],
                    "latex": "",
                    "category": f"Wikipedia/{category_hint}",
                }
            )
        processed = min(start + len(chunk), total_titles)
        if progress_file and len(items) % 100 == 0:
            save_items(items, progress_file)
        if processed % 100 == 0 or processed == total_titles:
            print(f"  processed titles: {processed}, collected facts: {len(items)}")
        if delay_s > 0:
            time.sleep(delay_s)
    if progress_file:
        save_items(items, progress_file)
    return items


def save_items(items: list[dict[str, str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=int,
        default=1000,
        help="Сколько валидных фактов собрать (по умолчанию: 1000)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.08,
        help="Пауза между батч-запросами в секундах (по умолчанию: 0.08)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Размер батча title-запроса к API (по умолчанию: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help="Путь output JSON",
    )
    args = parser.parse_args()

    print(f"Target facts: {args.target}")
    output_file = Path(args.output)
    items = build_wikipedia_facts(
        target=max(1, args.target),
        delay_s=max(0.0, args.delay),
        progress_file=output_file,
        batch_size=max(1, args.batch_size),
    )
    save_items(items, output_file)
    print(f"Saved {len(items)} facts -> {output_file}")


if __name__ == "__main__":
    main()
