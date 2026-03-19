"""
RAG-модуль: векторная база знаний для верификации математических доказательств.
Содержит загрузку данных, индексацию и поиск по определениям, аксиомам и теоремам.
"""

from app.core.clients.db.rag.config import RAGConfig
from app.core.clients.db.rag.loader import load_knowledge_from_path
from app.core.clients.db.rag.vector_store import VectorStore
from app.core.clients.db.rag.embedder import Embedder

__all__ = [
    "RAGConfig",
    "load_knowledge_from_path",
    "VectorStore",
    "Embedder",
]
