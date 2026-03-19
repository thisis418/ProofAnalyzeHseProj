"""
Загрузчик данных базы знаний из JSON-файлов.
Единая схема: type, name, statement, latex (опционально), category (опционально).
"""

from pathlib import Path
import json
from typing import Any

# Допустимые типы записей
KNOWLEDGE_TYPES = ("definition", "axiom", "theorem")


def _normalize_item(
    raw: dict[str, Any], source_file: str, index: int
) -> dict[str, Any] | None:
    """Приводит одну запись к единой схеме. Возвращает None при невалидных данных."""
    item_type = (raw.get("type") or "").strip().lower()
    if item_type not in KNOWLEDGE_TYPES:
        return None
    name = (raw.get("name") or "").strip()
    statement = (raw.get("statement") or "").strip()
    if not name and not statement:
        return None
    return {
        "type": item_type,
        "name": name or f"Untitled ({source_file}#{index})",
        "statement": statement,
        "latex": (raw.get("latex") or "").strip(),
        "category": (raw.get("category") or "").strip(),
    }


def load_json_file(file_path: Path) -> list[dict[str, Any]]:
    """
    Загружает один JSON-файл. Ожидает список объектов или один объект с полем 'items'.
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "items" in data:
        items = data["items"]
    else:
        items = [data] if isinstance(data, dict) else []
    source = file_path.name
    result = []
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue
        normalized = _normalize_item(raw, source, i)
        if normalized:
            result.append(normalized)
    return result


def load_knowledge_from_path(path: Path | str) -> list[dict[str, Any]]:
    """
    Загружает все записи из пути: если это файл — один JSON; если каталог — все .json в нём.
    """
    path = Path(path)
    if not path.exists():
        return []
    if path.is_file():
        if path.suffix.lower() != ".json":
            return []
        return load_json_file(path)
    # директория
    all_items = []
    for f in sorted(path.glob("*.json")):
        all_items.extend(load_json_file(f))
    return all_items
