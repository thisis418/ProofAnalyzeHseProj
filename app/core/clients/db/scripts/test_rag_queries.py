"""
Скрипт для ручной проверки работы векторной БД (RAG).
Запускает тестовые запросы и выводит топ результатов.

Перед запуском соберите БД:  python -m rag.build

Запуск:
  python scripts/test_rag_queries.py              — прогон всех тестовых запросов
  python scripts/test_rag_queries.py "ваш запрос" — один запрос, топ-5 результатов

На Windows при кракозябрах в консоли:  set PYTHONIOENCODING=utf-8  перед запуском
или запуск из IDE (PyCharm и т.д.) обычно выводит UTF-8 корректно.
"""

import os
import sys
from pathlib import Path

# Уменьшаем вывод sentence-transformers при первой загрузке модели
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import VectorStore


# Тестовые запросы: разные темы (геометрия, теоремы, определения)
TEST_QUERIES = [
    "сумма углов треугольника равна 180",
    "теорема Пифагора квадрат гипотенузы",
    "равные треугольники по двум сторонам и углу",
    "Kraft inequality prefix code",
    "Huffman coding compression",
    "channel capacity Shannon",
    "биссектриса угла медиана высота",
    "parallel lines axiom Euclid",
    "Markov inequality probability",
]


def shorten(text: str, max_len: int = 200) -> str:
    """Обрезает текст и добавляет ... при необходимости."""
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def run_test_queries(top_k: int = 3):
    store = VectorStore()
    collection = store._get_collection(create_if_missing=False)
    if collection is None:
        print(
            "Ошибка: векторная БД не найдена. Сначала выполните:  python -m rag.build"
        )
        return

    print("=" * 60)
    print("Проверка векторной БД: тестовые запросы")
    print("=" * 60)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n--- Запрос {i}: «{query}» ---")
        results = store.search(query, top_k=top_k)
        if not results:
            print("  (ничего не найдено)")
            continue
        for j, r in enumerate(results, 1):
            print(
                f"  {j}. [{r['type']}] {r['name']} (score: {r['score']}, category: {r['category']})"
            )
            print(f"     {shorten(r['statement'])}")
            if r.get("latex"):
                print(f"     LaTeX: {r['latex']}")

    print("\n" + "=" * 60)
    print("Готово. Проверьте, что в топе релевантные теоремы/определения.")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Свой запрос:  python scripts/test_rag_queries.py "ваш запрос"
        query = " ".join(sys.argv[1:])
        store = VectorStore()
        if store._get_collection(create_if_missing=False) is None:
            print(
                "Ошибка: векторная БД не найдена. Сначала выполните:  python -m rag.build"
            )
            sys.exit(1)
        print(f"Запрос: «{query}»\n")
        for j, r in enumerate(store.search(query, top_k=5), 1):
            print(
                f"{j}. [{r['type']}] {r['name']} (score: {r['score']}, category: {r['category']})"
            )
            print(f"   {shorten(r['statement'], 300)}\n")
    else:
        run_test_queries(top_k=3)
