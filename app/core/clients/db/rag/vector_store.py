"""
Векторное хранилище на ChromaDB: создание коллекции, добавление документов, поиск.
"""

from typing import Any

import chromadb
from chromadb.config import Settings

from app.core.clients.db.rag.config import RAGConfig
from app.core.clients.db.rag.embedder import Embedder, _make_chroma_embedding_function


def _document_to_text(doc: dict[str, Any]) -> str:
    """Собирает текст для эмбеддинга: название + формулировка (и LaTeX при наличии)."""
    parts = [doc.get("name", ""), doc.get("statement", "")]
    if doc.get("latex"):
        parts.append(doc["latex"])
    return " ".join(p for p in parts if p).strip() or " "


def _doc_to_metadata(doc: dict[str, Any]) -> dict[str, str]:
    """Метаданные для Chroma (только строки)."""
    return {
        "type": doc.get("type", ""),
        "name": doc.get("name", ""),
        "statement": doc.get("statement", ""),
        "latex": doc.get("latex", ""),
        "category": doc.get("category", ""),
    }


class VectorStore:
    """
    Векторное хранилище знаний. Персистентная коллекция ChromaDB + локальные эмбеддинги.
    """

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self._client = None
        self._collection = None
        self._embedder = Embedder(self.config.embedding_model)

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            path = str(self.config.chroma_persist_dir)
            self.config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self, create_if_missing: bool = True):
        client = self._get_client()
        try:
            coll = client.get_collection(name=self.config.collection_name)
        except Exception:
            if not create_if_missing:
                return None
            chroma_ef = _make_chroma_embedding_function(self._embedder)
            coll = client.create_collection(
                name=self.config.collection_name,
                embedding_function=chroma_ef,
                metadata={
                    "description": "Math knowledge: definitions, axioms, theorems"
                },
            )
        return coll

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """
        Добавляет документы в коллекцию. Если коллекция уже существует — пересоздаёт её
        и заполняет заново (для простоты пересборки БД).
        """
        client = self._get_client()
        try:
            client.delete_collection(self.config.collection_name)
        except Exception:
            pass
        chroma_ef = _make_chroma_embedding_function(self._embedder)
        collection = client.create_collection(
            name=self.config.collection_name,
            embedding_function=chroma_ef,
            metadata={"description": "Math knowledge"},
        )
        if not documents:
            return
        ids = []
        documents_texts = []
        metadatas = []
        for i, doc in enumerate(documents):
            ids.append(f"{doc.get('type', 'item')}_{i}")
            documents_texts.append(_document_to_text(doc))
            metadatas.append(_doc_to_metadata(doc))
        collection.add(
            ids=ids,
            documents=documents_texts,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Поиск по базе знаний. Возвращает список записей с полями:
        type, name, statement, latex, category, score.
        type_filter: опционально ограничить тип (definition / axiom / theorem).
        """
        collection = self._get_collection(create_if_missing=False)
        if collection is None:
            return []
        where = None
        if type_filter and type_filter.strip().lower() in (
            "definition",
            "axiom",
            "theorem",
        ):
            where = {"type": type_filter.strip().lower()}
        query_embedding = self._embedder.embed_query(query)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )
        if not result or not result["metadatas"] or not result["metadatas"][0]:
            return []
        # Chroma возвращает L2 distance; преобразуем в подобие score (меньше расстояние — выше релевантность)
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]
        out = []
        for meta, dist in zip(metadatas, distances):
            # score: чем меньше distance, тем лучше; нормализуем в [0,1]-подобное
            score = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
            out.append(
                {
                    "type": meta.get("type", ""),
                    "name": meta.get("name", ""),
                    "statement": meta.get("statement", ""),
                    "latex": meta.get("latex", ""),
                    "category": meta.get("category", ""),
                    "score": round(score, 4),
                }
            )
        return out
