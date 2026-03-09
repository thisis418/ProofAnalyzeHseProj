# RAG: векторная база знаний для верификации доказательств

Хранит определения, аксиомы и теоремы. По запросу (шаг доказательства) ищет релевантные формулировки в базе.

---

## Что за файлы

**rag/** — модуль базы знаний  
- `config.py` — пути к папкам, имя коллекции, модель эмбеддингов  
- `loader.py` — загрузка JSON из knowledge_base (поля: type, name, statement, latex, category)  
- `embedder.py` — векторизация текста через sentence-transformers  
- `vector_store.py` — ChromaDB: добавление документов, поиск по смыслу (search)  
- `build.py` — пересборка БД из всех JSON в knowledge_base  

**knowledge_base/** — исходные данные (редактируемые)  
- `euclidean_geometry.json` — определения, аксиомы, теоремы евклидовой геометрии  
- `theoremqa.json` — теоремы из датасета TheoremQA (скачивается скриптом)  

**scripts/**  
- `download_theoremqa.py` — скачать TheoremQA с GitHub и сохранить в knowledge_base/theoremqa.json  
- `test_rag_queries.py` — прогнать тестовые запросы к БД или один свой запрос  

**data/chroma_db/** — сгенерированное хранилище ChromaDB (создаётся при build)  

**requirements.txt** — chromadb, sentence-transformers, numpy  

---

## Эмбеддер

Используется **sentence-transformers** с моделью **`paraphrase-multilingual-MiniLM-L12-v2`**.

- **Multilingual** — в базе и русский (геометрия), и английский (TheoremQA). Одна модель обрабатывает оба языка, запрос можно писать на любом.
- **Локально и бесплатно** — всё крутится на своей машине, без API-ключей и лимитов. Удобно для учёбы и разработки.
- **Paraphrase** — модель заточена под семантическое сходство (перефразы, похожий смысл), а не под точное совпадение слов. Для поиска «обоснований» по шагу доказательства это подходит.
- **MiniLM-L12** — модель относительно лёгкая и быстрая, при этом качество для такого RAG достаточное.

Альтернативы не брали: чисто английские модели плохо работают с русской геометрией; облачные эмбеддинги (OpenAI и т.п.) — платно и лишняя зависимость.

---

## Скрипты для запуска

Установка зависимостей:
```bash
pip install -r requirements.txt
```

Скачать TheoremQA в knowledge_base (по желанию):
```bash
python scripts/download_theoremqa.py
```

Собрать или пересобрать векторную БД:
```bash
python -m rag.build
```

Проверить БД тестовыми запросами:
```bash
python scripts/test_rag_queries.py
```

Один свой запрос (топ-5):
```bash
python scripts/test_rag_queries.py "сумма углов треугольника"
```

В коде поиск: `from rag import VectorStore` → `VectorStore().search("запрос", top_k=5)`.
