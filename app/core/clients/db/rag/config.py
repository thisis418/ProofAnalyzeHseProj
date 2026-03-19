"""
Конфигурация RAG: пути, имя коллекции, модель эмбеддингов.
"""

from pathlib import Path

# Корень проекта (родитель папки rag)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Путь к каталогу с исходными данными (JSON)
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

# Путь к персистентному хранилищу ChromaDB
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma_db"

# Имя коллекции в ChromaDB
COLLECTION_NAME = "math_knowledge"

# Модель для эмбеддингов (multilingual для русского + формул в тексте)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RAGConfig:
    """Настраиваемая конфигурация RAG."""

    def __init__(
        self,
        knowledge_base_dir: Path | None = None,
        chroma_persist_dir: Path | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ):
        self.knowledge_base_dir = knowledge_base_dir or KNOWLEDGE_BASE_DIR
        self.chroma_persist_dir = chroma_persist_dir or CHROMA_PERSIST_DIR
        self.collection_name = collection_name or COLLECTION_NAME
        self.embedding_model = embedding_model or EMBEDDING_MODEL
