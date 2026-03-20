"""
Векторное хранилище на ChromaDB: создание коллекции, добавление документов, поиск.
"""

import re
from typing import Any

import chromadb
from chromadb.config import Settings

from app.core.clients.db.rag.config import RAGConfig
from app.core.clients.db.rag.embedder import Embedder, _make_chroma_embedding_function
from app.core.clients.db.rag.loader import load_knowledge_from_path


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
        self._knowledge_items_cache: list[dict[str, Any]] | None = None

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").lower().strip()
        text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _get_knowledge_items(self) -> list[dict[str, Any]]:
        if self._knowledge_items_cache is None:
            self._knowledge_items_cache = load_knowledge_from_path(
                self.config.knowledge_base_dir
            )
        return self._knowledge_items_cache

    def _lexical_search(
        self,
        query: str,
        top_k: int = 5,
        type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return []
        query_tokens = set(normalized_query.split())
        if not query_tokens:
            return []

        candidates: list[tuple[float, dict[str, Any]]] = []

        for item in self._get_knowledge_items():
            item_type = str(item.get("type", "")).strip().lower()
            if type_filter and item_type != type_filter.strip().lower():
                continue
            name = str(item.get("name", ""))
            statement = str(item.get("statement", ""))
            hay_name = self._normalize_text(name)
            hay_statement = self._normalize_text(statement)
            if not hay_name and not hay_statement:
                continue
            name_tokens = set(hay_name.split())
            st_tokens = set(hay_statement.split())
            overlap_name = len(query_tokens & name_tokens)
            overlap_statement = len(query_tokens & st_tokens)
            if (
                overlap_name == 0
                and overlap_statement == 0
                and normalized_query not in hay_name
                and normalized_query not in hay_statement
            ):
                continue
            score = 0.0
            if normalized_query in hay_name:
                score += 3.0
            if normalized_query in hay_statement:
                score += 1.0
            score += overlap_name * 0.6 + overlap_statement * 0.2
            if "theorem" in normalized_query and "theorem" in hay_name:
                score += 0.8

            candidates.append(
                (
                    score,
                    {
                        "type": item_type,
                        "name": name,
                        "statement": statement,
                        "latex": str(item.get("latex", "")),
                        "category": str(item.get("category", "")),
                        # Bring lexical scores to the same [0,1]-like scale used in API.
                        "score": round(min(0.99, 0.4 + score / 12.0), 4),
                    },
                )
            )
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in candidates[:top_k]]

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
        out: list[dict[str, Any]] = []
        collection = self._get_collection(create_if_missing=False)
        where = None
        if type_filter and type_filter.strip().lower() in (
            "definition",
            "axiom",
            "theorem",
        ):
            where = {"type": type_filter.strip().lower()}

        if collection is not None:
            try:
                query_embedding = self._embedder.embed_query(query)
                result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where,
                    include=["metadatas", "distances"],
                )
                if result and result.get("metadatas") and result["metadatas"][0]:
                    distances = result["distances"][0]
                    metadatas = result["metadatas"][0]
                    for meta, dist in zip(metadatas, distances):
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
            except Exception:
                # ANN retrieval can fail for index reasons; lexical fallback below.
                out = []

        lexical = self._lexical_search(query=query, top_k=top_k, type_filter=type_filter)
        if not out:
            return lexical

        merged: dict[str, dict[str, Any]] = {}
        for item in out + lexical:
            key = f"{item.get('type','')}|{item.get('name','')}"
            prev = merged.get(key)
            if prev is None or float(item.get("score", 0.0)) > float(prev.get("score", 0.0)):
                merged[key] = item
        final = sorted(merged.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return final[:top_k]
