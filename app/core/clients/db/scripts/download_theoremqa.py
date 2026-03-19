"""
Скачивает TheoremQA all_theorems.json и конвертирует в формат knowledge_base.
Запуск: python scripts/download_theoremqa.py
"""

import json
import urllib.request
from pathlib import Path

THEOREMQA_URL = (
    "https://raw.githubusercontent.com/wenhuchen/TheoremQA/main/all_theorems.json"
)
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
OUTPUT_FILE = KNOWLEDGE_BASE_DIR / "theoremqa.json"


def main():
    print("Downloading TheoremQA all_theorems.json...")
    with urllib.request.urlopen(THEOREMQA_URL, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    items = []
    for name, statement in raw.items():
        if not name or not statement or not isinstance(statement, str):
            continue
        name = name.strip()
        statement = statement.strip()
        if not name or not statement:
            continue
        items.append(
            {
                "type": "theorem",
                "name": name,
                "statement": statement,
                "latex": "",
                "category": "TheoremQA",
            }
        )

    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(items)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
