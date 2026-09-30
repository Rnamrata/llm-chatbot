import sys
import pytest
from typing import Any, List, Optional
from unittest.mock import MagicMock

from langchain_core.language_models.llms import LLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


def _purge_app_modules():
    """
    Drop every project module (main + everything under src) so the next
    import re-reads env vars and re-applies whatever monkeypatches are
    active for this test. Needed because main.py builds its singletons
    (vector_store, file_manager, chat_manager, ...) once at import time.
    """
    for name in list(sys.modules):
        if name == 'main' or name == 'src' or name.startswith('src.'):
            del sys.modules[name]


class FakeEmbeddings:
    """Stand-in for langchain_ollama.OllamaEmbeddings — no network calls"""
    def __init__(self, *args, **kwargs):
        pass

    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 8


class FakeOllamaLLM(LLM):
    """
    Stand-in for langchain_ollama.OllamaLLM. Subclasses the real LangChain
    LLM base class (not just duck-typed) because ConversationalRetrievalChain
    validates its `llm` field with Pydantic and rejects arbitrary objects.
    """
    model: str = ""
    temperature: float = 0.0
    canned_response: str = "This is a fake LLM response."

    @property
    def _llm_type(self) -> str:
        return "fake-ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None, **kwargs: Any) -> str:
        return self.canned_response


class _FakeRetriever(BaseRetriever):
    """Real BaseRetriever subclass backing FakeChroma.as_retriever()"""
    store: Any = None
    k: int = 4
    filter: Any = None

    def _get_relevant_documents(self, query, *, run_manager=None):
        return self.store.max_marginal_relevance_search(query, k=self.k, filter=self.filter)


class _FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def count(self):
        return len(self._docs)


class FakeChroma:
    """In-memory stand-in for langchain_chroma.Chroma — no real Chroma/SQLite"""

    def __init__(self, collection_name=None, embedding_function=None, persist_directory=None):
        self._docs = []  # list of (id, page_content, metadata)
        self._collection = _FakeCollection(self._docs)

    def _matches(self, metadata, where):
        if not where:
            return True
        return all(metadata.get(k) == v for k, v in where.items())

    def add_documents(self, documents):
        ids = []
        for doc in documents:
            doc_id = str(len(self._docs))
            self._docs.append((doc_id, doc.page_content, doc.metadata))
            ids.append(doc_id)
        return ids

    def max_marginal_relevance_search(self, query, k=4, fetch_k=20, lambda_mult=0.5, filter=None, **kwargs):
        matched = [
            Document(page_content=text, metadata=meta)
            for _id, text, meta in self._docs
            if self._matches(meta, filter)
        ]
        return matched[:k]

    def as_retriever(self, search_kwargs=None):
        search_kwargs = search_kwargs or {}
        return _FakeRetriever(store=self, k=search_kwargs.get('k', 4), filter=search_kwargs.get('filter'))

    def delete(self, ids=None, where=None):
        self._docs[:] = [d for d in self._docs if not self._matches(d[2], where)]

    def get(self, ids=None, where=None, include=None):
        matched = [d for d in self._docs if self._matches(d[2], where)]
        return {
            'ids': [d[0] for d in matched],
            'documents': [d[1] for d in matched],
            'metadatas': [d[2] for d in matched],
        }


@pytest.fixture
def fake_review_client():
    review_client = MagicMock()
    review_client.review.return_value = {'success': True, 'findings': []}
    review_client.health.return_value = {'success': True, 'status': 'ok'}
    return review_client


@pytest.fixture
def app(monkeypatch, tmp_path, fake_review_client):
    """Flask app with Ollama, Chroma, and ReviewClient mocked; I/O isolated to tmp_path"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CHROMA_PERSIST_DIR', str(tmp_path / 'chroma_db'))
    monkeypatch.setenv('UPLOAD_DIR', str(tmp_path / 'uploads'))
    monkeypatch.setenv('MEDIA_DIR', str(tmp_path / 'uploads' / 'media'))

    _purge_app_modules()

    monkeypatch.setattr('src.modules.vector_store_and_embedding.OllamaEmbeddings', FakeEmbeddings)
    monkeypatch.setattr('src.modules.vector_store_and_embedding.Chroma', FakeChroma)
    monkeypatch.setattr('src.modules.llm_manager.OllamaLLM', FakeOllamaLLM)
    monkeypatch.setattr('src.modules.review_client.ReviewClient', lambda *a, **kw: fake_review_client)

    import main as main_module
    main_module.app.config.update(TESTING=True)

    yield main_module.app

    _purge_app_modules()


@pytest.fixture
def client(app):
    return app.test_client()