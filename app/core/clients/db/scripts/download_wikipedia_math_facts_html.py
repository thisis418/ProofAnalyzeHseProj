"""
Собирает математические факты из Wikipedia через HTML-страницы категорий
(без MediaWiki API), чтобы избежать API rate-limit.

Запуск:
  python app/core/clients/db/scripts/download_wikipedia_math_facts_html.py --target 1000
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

BASE_URL = "https://en.wikipedia.org"

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
OUTPUT_FILE = KNOWLEDGE_BASE_DIR / "wikipedia_math_facts.json"

SEED_CATEGORIES = [
    "/wiki/Category:Mathematics",
    "/wiki/Category:Mathematical_theorems",
    "/wiki/Category:Number_theory",
    "/wiki/Category:Geometry",
    "/wiki/Category:Mathematical_analysis",
    "/wiki/Category:Linear_algebra",
    "/wiki/Category:Combinatorics",
    "/wiki/Category:Probability_theory",
    "/wiki/Category:Topology",
    "/wiki/Category:Graph_theory",
    "/wiki/Category:Set_theory",
    "/wiki/Category:Mathematical_logic",
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
    "principle",
)
AXIOM_HINTS = ("axiom", "postulate")


def _download_html(url: str, timeout: int = 30) -> str | None:
    req = Request(
        url,
        headers={
            "User-Agent": "ProofsAnalyzeProj-RAG-Bot/1.0 (academic project)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    backoff = 1.0
    for attempt in range(1, 6):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 20.0)
                continue
            return None
        except URLError:
            if attempt < 5:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 20.0)
                continue
            return None
    return None


def _is_article_href(href: str) -> bool:
    if not href or not href.startswith("/wiki/"):
        return False
    bad_prefixes = (
        "/wiki/Category:",
        "/wiki/Help:",
        "/wiki/File:",
        "/wiki/Talk:",
        "/wiki/Template:",
        "/wiki/Wikipedia:",
        "/wiki/Portal:",
        "/wiki/Special:",
    )
    if any(href.startswith(p) for p in bad_prefixes):
        return False
    return True


def _iter_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for val in values:
        key = val.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _parse_category_page(
    html: str,
) -> tuple[list[str], list[str], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    page_links: list[str] = []
    subcats: list[str] = []
    next_page: str | None = None

    subcats_div = soup.find("div", id="mw-subcategories")
    if subcats_div:
        for a in subcats_div.select("a[href]"):
            href = a.get("href", "")
            if href.startswith("/wiki/Category:"):
                subcats.append(href)

    pages_div = soup.find("div", id="mw-pages")
    if pages_div:
        for a in pages_div.select("a[href]"):
            href = a.get("href", "")
            if _is_article_href(href):
                page_links.append(href)
        next_a = pages_div.select_one("a.mw-nextlink[href]")
        if next_a:
            href = next_a.get("href", "")
            if href.startswith("/w/index.php?"):
                next_page = href

    return _iter_unique(page_links), _iter_unique(subcats), next_page


def _extract_intro(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.find("h1", id="firstHeading") or soup.title).get_text(" ", strip=True)
    content = soup.find("div", class_="mw-parser-output")
    if not content:
        return title, ""
    paragraphs: list[str] = []
    for p in content.find_all("p", recursive=False):
        txt = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if txt:
            paragraphs.append(txt)
        if len(" ".join(paragraphs)) > 900:
            break
    statement = " ".join(paragraphs).strip()
    if len(statement) > 1000:
        statement = statement[:1000].rsplit(" ", 1)[0] + "..."
    return title, statement


def _infer_type(name: str) -> str:
    lower = name.lower()
    if any(h in lower for h in AXIOM_HINTS):
        return "axiom"
    if any(h in lower for h in THEOREM_HINTS):
        return "theorem"
    return "definition"


def collect_titles(max_titles: int, max_categories: int, delay_s: float) -> list[str]:
    queue = deque(SEED_CATEGORIES)
    visited_categories: set[str] = set()
    titles: list[str] = []
    visited_pages: set[str] = set()

    while queue and len(visited_categories) < max_categories and len(titles) < max_titles:
        cat = queue.popleft()
        if cat in visited_categories:
            continue
        visited_categories.add(cat)

        next_page = cat
        while next_page and len(titles) < max_titles:
            html = _download_html(urljoin(BASE_URL, next_page))
            if not html:
                break
            page_links, subcats, nxt = _parse_category_page(html)
            for sub in subcats:
                if sub not in visited_categories and len(queue) < max_categories * 2:
                    queue.append(sub)
            for link in page_links:
                if link in visited_pages:
                    continue
                visited_pages.add(link)
                titles.append(link)
                if len(titles) >= max_titles:
                    break
            next_page = nxt
            if delay_s > 0:
                time.sleep(delay_s)

    return titles


def build_facts(target: int, delay_s: float) -> list[dict[str, str]]:
    candidate_target = max(target * 2, target + 500)
    titles = collect_titles(
        max_titles=candidate_target,
        max_categories=220,
        delay_s=delay_s,
    )
    print(f"Collected candidate titles: {len(titles)}")
    items: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for idx, href in enumerate(titles, 1):
        if len(items) >= target:
            break
        html = _download_html(urljoin(BASE_URL, href))
        if not html:
            continue
        name, statement = _extract_intro(html)
        if not name or len(statement) < 80:
            continue
        key = name.casefold()
        if key in seen_names:
            continue
        seen_names.add(key)
        items.append(
            {
                "type": _infer_type(name),
                "name": name,
                "statement": statement,
                "latex": "",
                "category": "Wikipedia/Mathematics",
            }
        )
        if idx % 100 == 0:
            print(f"  processed pages: {idx}, collected facts: {len(items)}")
        if delay_s > 0:
            time.sleep(delay_s)
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--delay", type=float, default=0.06)
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE))
    args = parser.parse_args()

    print(f"Target facts: {args.target}")
    facts = build_facts(target=max(1, args.target), delay_s=max(0.0, args.delay))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(facts)} facts -> {output}")


if __name__ == "__main__":
    main()
