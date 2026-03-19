"""
Скрипт построения векторной базы знаний из файлов в knowledge_base/.
Запуск: python -m rag.build или из корня проекта.
"""

from pathlib import Path

from app.core.clients.db.rag.config import RAGConfig
from app.core.clients.db.rag.loader import load_knowledge_from_path
from app.core.clients.db.rag.vector_store import VectorStore


def build_knowledge_base(
    knowledge_base_dir: Path | None = None,
    config: RAGConfig | None = None,
) -> int:
    """
    Загружает данные из knowledge_base_dir, индексирует в ChromaDB.
    Возвращает количество проиндексированных записей.
    """
    cfg = config or RAGConfig()
    kb_dir = knowledge_base_dir or cfg.knowledge_base_dir
    if not kb_dir.exists():
        kb_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Создана пустая папка {kb_dir}. Добавьте JSON-файлы с определениями, аксиомами и теоремами."
        )
        return 0
    documents = load_knowledge_from_path(kb_dir)
    if not documents:
        print(
            f"В {kb_dir} не найдено валидных записей (ожидаются JSON с полями type, name, statement)."
        )
        return 0
    store = VectorStore(cfg)
    store.add_documents(documents)
    cfg.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Проиндексировано записей: {len(documents)}. База сохранена в {cfg.chroma_persist_dir}"
    )
    return len(documents)


def main():
    build_knowledge_base()


if __name__ == "__main__":
    main()
