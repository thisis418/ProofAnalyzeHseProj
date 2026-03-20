"""
Импортирует структурированные факты из OEIS в формат knowledge_base.

Поддерживаемые источники:
1) gz-дампы OEIS (names.gz + stripped.gz)
2) локальный клон oeisdata (Git export)

Запуск:
  python app/core/clients/db/scripts/download_oeis.py --source gz --limit 5000
  python app/core/clients/db/scripts/download_oeis.py --source repo --repo-path /path/to/oeisdata --limit 5000
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

NAMES_GZ_URL = "https://oeis.org/names.gz"
STRIPPED_GZ_URL = "https://oeis.org/stripped.gz"
JSON_API_URL = "https://oeis.org/search?fmt=json&q=id:{aid}"

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
OUTPUT_FILE = KNOWLEDGE_BASE_DIR / "oeis_facts.json"

USER_AGENT = "ProofsAnalyzeProj-OEIS-Importer/1.0"


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _iter_gzip_lines(raw_bytes: bytes) -> Iterable[str]:
    with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as gz:
        text = gz.read().decode("utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield line


def _parse_names_gz() -> dict[str, str]:
    print("Downloading names.gz...")
    raw = _download_bytes(NAMES_GZ_URL)
    result: dict[str, str] = {}
    for line in _iter_gzip_lines(raw):
        if line.startswith("#"):
            continue
        if not line.startswith("A"):
            continue
        m = re.match(r"^(A\d{6})\s+(.+)$", line)
        if not m:
            continue
        aid = m.group(1)
        desc = m.group(2).strip()
        if desc:
            result[aid] = desc
    return result


def _parse_stripped_gz(max_terms: int = 40) -> dict[str, list[str]]:
    print("Downloading stripped.gz...")
    raw = _download_bytes(STRIPPED_GZ_URL)
    result: dict[str, list[str]] = {}
    for line in _iter_gzip_lines(raw):
        if line.startswith("#"):
            continue
        if not line.startswith("A"):
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        aid = parts[0]
        seq_raw = parts[1].strip()
        seq_raw = seq_raw.lstrip(",")
        nums = [x.strip() for x in seq_raw.split(",") if x.strip()]
        if max_terms > 0:
            nums = nums[:max_terms]
        result[aid] = nums
    return result


def _parse_repo_seq_file(path: Path, max_terms: int = 40) -> dict[str, str | list[str]] | None:
    aid = path.stem
    if not re.fullmatch(r"A\d{6}", aid):
        return None
    desc = ""
    chunks: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for line in text.splitlines():
        if line.startswith("%N"):
            # %N A000045 Fibonacci numbers.
            parts = line.split(" ", 2)
            if len(parts) >= 3 and not desc:
                desc = parts[2].strip()
        if line.startswith("%S") or line.startswith("%T") or line.startswith("%U"):
            parts = line.split(" ", 2)
            if len(parts) >= 3:
                chunks.append(parts[2].strip())
    terms: list[str] = []
    if chunks:
        joined = ",".join(chunks)
        terms = [x.strip() for x in joined.split(",") if x.strip()]
    if max_terms > 0:
        terms = terms[:max_terms]
    if not desc and not terms:
        return None
    return {"aid": aid, "desc": desc, "terms": terms}


def _parse_oeis_repo(repo_path: Path, max_terms: int = 40) -> tuple[dict[str, str], dict[str, list[str]]]:
    seq_root = repo_path / "seq"
    if not seq_root.exists():
        raise FileNotFoundError(f"Expected seq directory in {repo_path}")
    names: dict[str, str] = {}
    terms: dict[str, list[str]] = {}
    count = 0
    for p in seq_root.rglob("*.seq"):
        parsed = _parse_repo_seq_file(p, max_terms=max_terms)
        if not parsed:
            continue
        aid = str(parsed["aid"])
        desc = str(parsed["desc"])
        seq_terms = list(parsed["terms"])
        if desc:
            names[aid] = desc
        terms[aid] = seq_terms
        count += 1
        if count % 50000 == 0:
            print(f"  parsed repo entries: {count}")
    print(f"Parsed repo entries total: {count}")
    return names, terms


def _fetch_json_api_desc(aid: str) -> str:
    url = JSON_API_URL.format(aid=urllib.parse.quote(aid))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    results = payload.get("results", [])
    if not results:
        return ""
    name = str(results[0].get("name", "")).strip()
    return name


def _build_items(
    names: dict[str, str],
    terms: dict[str, list[str]],
    limit: int,
) -> list[dict[str, str]]:
    aids = sorted(set(names) | set(terms))
    items: list[dict[str, str]] = []
    for aid in aids:
        desc = names.get(aid, "").strip()
        seq = terms.get(aid, [])
        if not desc and not seq:
            continue
        seq_text = ", ".join(seq) if seq else "n/a"
        statement_parts = []
        if desc:
            statement_parts.append(desc)
        statement_parts.append(f"First terms: {seq_text}.")
        statement_parts.append(f"OEIS id: {aid}.")
        statement_parts.append(f"Source: https://oeis.org/{aid}")
        statement = " ".join(statement_parts).strip()
        name = f"{aid} {desc}".strip() if desc else aid
        items.append(
            {
                "type": "definition",
                "name": name,
                "statement": statement,
                "latex": "",
                "category": "OEIS",
            }
        )
        if 0 < limit <= len(items):
            break
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["gz", "repo", "json-api"],
        default="gz",
        help="Источник данных: gz | repo | json-api",
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default="",
        help="Путь к локальному клону oeisdata (для --source repo)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Сколько фактов сохранить (по умолчанию 5000)",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=40,
        help="Сколько первых членов последовательности сохранять (по умолчанию 40)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help="Куда сохранить JSON",
    )
    args = parser.parse_args()

    limit = max(1, args.limit)
    max_terms = max(1, args.max_terms)

    if args.source == "gz":
        names = _parse_names_gz()
        terms = _parse_stripped_gz(max_terms=max_terms)
    elif args.source == "repo":
        if not args.repo_path:
            raise ValueError("--repo-path is required for --source repo")
        names, terms = _parse_oeis_repo(Path(args.repo_path), max_terms=max_terms)
    else:
        # json-api режим: берём список id из names.gz и подтягиваем описания через JSON API
        # (полезно для валидации API, но медленнее bulk-дампов)
        base_names = _parse_names_gz()
        terms = _parse_stripped_gz(max_terms=max_terms)
        names = {}
        for i, aid in enumerate(sorted(base_names)[:limit], 1):
            try:
                names[aid] = _fetch_json_api_desc(aid) or base_names[aid]
            except Exception:
                names[aid] = base_names[aid]
            if i % 200 == 0:
                print(f"  json-api fetched: {i}")

    items = _build_items(names=names, terms=terms, limit=limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
