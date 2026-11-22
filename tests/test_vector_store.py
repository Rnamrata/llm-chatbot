import io

import pytest
from langchain_core.documents import Document


def _docs(n):
    return [Document(page_content=f'chunk {i}', metadata={'source': 'book.pdf', 'type': 'file_upload'}) for i in range(n)]


@pytest.fixture
def store(app, monkeypatch):
    """The app's vector store (FakeChroma underneath), with retries that don't wait"""
    import main
    monkeypatch.setattr('src.modules.vector_store_and_embedding.time.sleep', lambda s: None)
    return main.vector_store


def _stored(store):
    return len(store.vectorstore.get()['ids'])


def _flaky(store, monkeypatch, fail_calls):
    """Makes the given add_documents calls (1-based) fail like a dropped Ollama connection"""
    real_add = store.vectorstore.add_documents
    calls = []

    def add(documents):
        calls.append(len(documents))
        if len(calls) in fail_calls:
            raise ValueError('Post "http://127.0.0.1:52804/tokenize": EOF (status code: 400)')
        return real_add(documents)

    monkeypatch.setattr(store.vectorstore, 'add_documents', add)
    return calls


def test_a_large_file_is_embedded_in_batches(store, monkeypatch):
    calls = _flaky(store, monkeypatch, fail_calls=set())

    result = store.store_chunks(_docs(150))

    assert result['count'] == 150
    assert calls == [64, 64, 22]
    assert _stored(store) == 150


def test_a_batch_that_fails_once_is_retried(store, monkeypatch):
    calls = _flaky(store, monkeypatch, fail_calls={2})

    result = store.store_chunks(_docs(150))

    assert result['count'] == 150
    assert calls == [64, 64, 64, 22]  # the second batch went twice
    assert _stored(store) == 150


def test_a_batch_that_keeps_failing_leaves_nothing_behind(store, monkeypatch):
    calls = _flaky(store, monkeypatch, fail_calls={2, 3, 4})

    with pytest.raises(ValueError, match='EOF'):
        store.store_chunks(_docs(150))

    assert calls == [64, 64, 64, 64]  # first batch, then three tries of the second
    assert _stored(store) == 0  # the first batch was removed again


def test_an_upload_survives_one_dropped_connection(client, store, monkeypatch):
    _flaky(store, monkeypatch, fail_calls={1})
    text = ('Gradient descent lowers the loss step by step. ' * 30 + '\n\n') * 3

    response = client.post(
        '/upload/file',
        data={'file': (io.BytesIO(text.encode()), 'notes.txt')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 200
    assert response.get_json()['success'] is True
    assert [s['name'] for s in client.get('/sources').get_json()['sources']] == ['notes.txt']

def test_a_pdf_with_a_broken_maths_character_can_be_added(client, store, monkeypatch):
    # What PyPDF2 returned for one page of a 413-page book: a maths letter cut in half
    monkeypatch.setattr(
        'src.modules.document_processor.DocumentProcessor.pdfToText',
        lambda self, path, name: 'The loss is ∥x−dφμx+σx\ud835∥2 for each sample.\n' * 40,
    )
    # The real Ollama client sends text as UTF-8, which is where a half character failed
    real_add = store.vectorstore.add_documents

    def add(documents):
        for doc in documents:
            doc.page_content.encode('utf-8')
        return real_add(documents)

    monkeypatch.setattr(store.vectorstore, 'add_documents', add)

    response = client.post(
        '/upload/file',
        data={'file': (io.BytesIO(b'%PDF-1.4 stand-in'), 'book.pdf')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 200, response.get_json()
    assert response.get_json()['success'] is True