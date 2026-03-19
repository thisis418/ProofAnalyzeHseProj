"""
Эмбеддинг-модуль на базе sentence-transformers.
Используется для векторизации формулировок и запросов при поиске по базе знаний.
"""

from typing import List


def _make_chroma_embedding_function(embedder: "Embedder"):
    """
    Возвращает экземпляр класса, совместимого с ChromaDB EmbeddingFunction.
    Chroma передаёт список текстов (или документов), возвращаем список векторов.
    """
    try:
        from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
    except ImportError:
        return _ChromaEmbeddingFunctionFallback(embedder)

    class _ChromaEmbeddingFunction(EmbeddingFunction):
        def __init__(self, ef):
            self._ef = ef

        def __call__(self, input: Documents) -> Embeddings:
            # Chroma может передать List[str] или список объектов с полем text
            texts = []
            for x in input:
                if isinstance(x, str):
                    texts.append(x)
                else:
                    texts.append(getattr(x, "text", str(x)))
            if not texts:
                return []
            return self._ef.embed_documents(texts)

    return _ChromaEmbeddingFunction(embedder)


class _ChromaEmbeddingFunctionFallback:
    """Простая обёртка, если chromadb.api.types недоступен (для тестов)."""

    def __init__(self, embedder: "Embedder"):
        self._embedder = embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        texts = [x if isinstance(x, str) else getattr(x, "text", str(x)) for x in input]
        return self._embedder.embed_documents(texts)


class Embedder:
    """Обёртка над sentence-transformers для генерации эмбеддингов."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Векторизует список текстов. Возвращает список векторов."""
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Векторизует один запрос (для поиска)."""
        return self.embed_documents([text])[0]
