"""
Очистка oeis_facts.json от служебных/малосодержательных записей.

Запуск:
  python app/core/clients/db/scripts/clean_oeis_facts.py
  python app/core/clients/db/scripts/clean_oeis_facts.py --input ... --output ...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent / "knowledge_base" / "oeis_facts.json"
)

ID_PREFIX_RE = re.compile(r"^A\d{6}\s*")
NON_SUBSTANTIVE_RE = [
    re.compile(r"\berroneous\b", re.IGNORECASE),
    re.compile(r"^\s*duplicate of\b", re.IGNORECASE),
    re.compile(r"^\s*cf\.\b", re.IGNORECASE),
    re.compile(r"\breserved\b", re.IGNORECASE),
    re.compile(r"\bdeprecated\b", re.IGNORECASE),
    re.compile(r"^\s*former\b", re.IGNORECASE),
]


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_desc(name: str) -> str:
    return _clean_space(ID_PREFIX_RE.sub("", name or ""))


def _is_substantive(desc: str) -> bool:
    if not desc:
        return False
    if len(desc) < 8:
        return False
    for rx in NON_SUBSTANTIVE_RE:
        if rx.search(desc):
            return False
    return True


def _extract_aid(name: str, statement: str) -> str:
    m = re.match(r"^(A\d{6})", (name or "").strip())
    if m:
        return m.group(1)
    m2 = re.search(r"\b(A\d{6})\b", statement or "")
    return m2.group(1) if m2 else ""


def _extract_terms(statement: str) -> str:
    m = re.search(
        r"First terms:\s*(.*?)\.\s*OEIS id:",
        statement or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return ""
    return _clean_space(m.group(1))


def _build_statement(desc: str, aid: str, terms: str) -> str:
    parts = [desc.rstrip(".") + "."]
    if terms:
        parts.append(f"First terms: {terms}.")
    return " ".join(parts)


def clean_items(items: list[dict]) -> tuple[list[dict], int]:
    cleaned: list[dict] = []
    dropped = 0
    seen: set[tuple[str, str]] = set()

    for item in items:
        name = str(item.get("name", "")).strip()
        statement = str(item.get("statement", "")).strip()
        desc = _extract_desc(name)
        if not _is_substantive(desc):
            dropped += 1
            continue
        aid = _extract_aid(name, statement)
        terms = _extract_terms(statement)
        canonical_name = desc
        key = (canonical_name.casefold(), terms.casefold())
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        cleaned.append(
            {
                "type": "definition",
                "name": canonical_name,
                "statement": _build_statement(desc=desc, aid=aid, terms=terms),
                "latex": "",
                "category": "OEIS",
            }
        )
    return cleaned, dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input file should contain a JSON list.")

    cleaned, dropped = clean_items(data)
    output_path.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Input: {len(data)}")
    print(f"Dropped: {dropped}")
    print(f"Saved: {len(cleaned)} -> {output_path}")


if __name__ == "__main__":
    main()
