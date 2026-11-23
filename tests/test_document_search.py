import io
from typing import Any, List
from unittest.mock import MagicMock

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from tests.conftest import FakeOllamaLLM


class RecordingLLM(FakeOllamaLLM):
    """Keeps every prompt it was sent"""
    prompts: List[str] = []

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        return self.canned_response


def _store():
    import main
    return main.vector_store.vectorstore


def _add(text, **metadata):
    _store().add_documents([Document(page_content=text, metadata=metadata)])


# ---------- embedding prefixes ----------

def test_passages_and_questions_get_their_nomic_prefixes():
    from src.modules.vector_store_and_embedding import PrefixedEmbeddings

    base = MagicMock()
    embeddings = PrefixedEmbeddings(base, document_prefix='search_document: ', query_prefix='search_query: ')
    embeddings.embed_documents(['a passage'])
    embeddings.embed_query('a question')

    base.embed_documents.assert_called_once_with(['search_document: a passage'])
    base.embed_query.assert_called_once_with('search_query: a question')


def test_the_prefixes_are_on_by_default(app):
    import main
    from src import config

    assert config.EMBED_DOCUMENT_PREFIX == 'search_document: '
    assert config.EMBED_QUERY_PREFIX == 'search_query: '
    assert main.vector_store.embeddings.document_prefix == config.EMBED_DOCUMENT_PREFIX


def test_reembed_replaces_every_vector_under_the_same_id():
    from scripts.reembed import reembed

    store = MagicMock()
    store.vectorstore._collection.get.return_value = {
        'ids': ['1', '2', '3'], 'documents': ['a', 'b', 'c'], 'metadatas': [{'type': 'web'}, {}, {'type': 'youtube'}],
    }
    store.embeddings.embed_documents.side_effect = lambda texts: [[0.5]] * len(texts)

    count = reembed(store, batch_size=2, log=lambda *_: None)

    upserts = store.vectorstore._collection.upsert.call_args_list
    assert count == 3
    assert [c.kwargs['ids'] for c in upserts] == [['1', '2'], ['3']]
    assert upserts[1].kwargs['documents'] == ['c'] and upserts[1].kwargs['metadatas'] == [{'type': 'youtube'}]


def test_a_dry_run_changes_nothing():
    from scripts.reembed import reembed

    store = MagicMock()
    store.vectorstore._collection.get.return_value = {'ids': ['1'], 'documents': ['a'], 'metadatas': [{}]}

    assert reembed(store, batch_size=8, dry_run=True, log=lambda *_: None) == 1
    store.vectorstore._collection.upsert.assert_not_called()


# ---------- what a document chat searches ----------

def test_a_document_chat_reads_8_passages_by_default(client):
    import main
    for i in range(10):
        _add(f'passage {i}', source='notes.pdf', type='file_upload')

    result = client.post('/chat', json={'question': 'What is in my notes?', 'session_id': 'd1'}).get_json()

    assert result['num_sources'] == 8
    assert main.chat_manager.sessions['d1']['k'] == 8


def test_a_document_chat_never_searches_code_reviews_or_unlabelled_chunks(client):
    _add('Gradient descent lowers the loss.', source='notes.pdf', type='file_upload')
    _add('def find_product(self, name):', source='inventory_service.py', type='code_review', review_id='r1')
    _add('inventory_service.py, line 20 (high): SQL injection', source='inventory_service.py', type='review_finding')
    _add('An old chunk with no details')

    result = client.post('/chat', json={'question': 'What is gradient descent?', 'session_id': 'd1'}).get_json()

    assert [s['metadata']['source'] for s in result['sources']] == ['notes.pdf']


def test_every_passage_is_shown_to_the_model_with_its_source(client):
    import main
    llm = RecordingLLM(prompts=[])
    main.llm_manager.llm = llm
    _add('Gradient descent lowers the loss.', source='notes.pdf', type='file_upload', added_at='2026-11-20T10:00:00+00:00')
    _add('Neural networks learn weights.', source='https://youtu.be/x', type='youtube',
         filename='What is a Neural Network_transcription.txt', added_at='2026-11-21T09:00:00+00:00')
    _add('Trust is inverted.', source='https://example.com/t', type='web', title='The trust inversion')

    client.post('/chat', json={'question': 'Which sources do I have?', 'session_id': 'd1'})

    prompt = llm.prompts[-1]
    assert '[notes.pdf (PDF, added 2026-11-20 10:00 UTC)]\nGradient descent lowers the loss.' in prompt
    assert '[What is a Neural Network (YouTube video, added 2026-11-21 09:00 UTC)]\nNeural networks learn weights.' in prompt
    assert '[The trust inversion (web page)]\nTrust is inverted.' in prompt


def test_sources_sent_to_lens_keep_their_own_text(client):
    _add('Gradient descent lowers the loss.', source='notes.pdf', type='file_upload')

    result = client.post('/chat', json={'question': 'What is gradient descent?', 'session_id': 'd1'}).get_json()

    assert result['sources'][0]['content'] == 'Gradient descent lowers the loss.'


def test_source_labels_name_the_kind_of_file():
    from src.modules.document_context import source_label

    assert source_label({'type': 'file_upload', 'source': 'Notes.MD'}) == 'Notes.MD (Markdown file)'
    assert source_label({'type': 'web', 'source': 'u', 'added_at': '2026-01-02'}) == 'u (web page, added 2026-01-02)'
    assert source_label({'type': 'file_upload', 'source': 'a.txt', 'added_at': '2026-01-02T03:04:05'}) == 'a.txt (text file, added 2026-01-02 03:04 UTC)'


# ---------- web pages ----------

PAGE = """<html lang="en"><head><title>The trust inversion | Institute</title>
<meta name="description" content="Why trust flips"></head><body>
<header class="site-header"><a href="/">Home</a><a href="/all">All articles</a></header>
<nav><a>Become a member</a></nav>
<div class="cookie-banner">We use cookies</div>
<section class="article-hero"><div class="hero__download-cta"><a>Download article</a></div>
<h1>The trust inversion</h1><p>Oct 27, 2026 | 7 min read</p></section>
<div class="article-body"><p>The danger in 2026 is that people trust machines more than each other.</p>
<p>Regulators   are   responding.</p>
<aside><p>Related: other articles</p></aside>
<section class="article-archives"><h2>You might also like...</h2></section></div>
<script>var x = 1;</script>
<footer><p>Copyright 2026</p></footer>
</body></html>"""


def test_a_web_page_keeps_its_article_and_drops_the_site_around_it():
    from src.modules.document_processor import clean_web_page

    title, text = clean_web_page(BeautifulSoup(PAGE, 'html.parser'))

    assert title == 'The trust inversion | Institute'
    assert text.split('\n') == [
        'The trust inversion',
        'Oct 27, 2026 | 7 min read',
        'The danger in 2026 is that people trust machines more than each other.',
        'Regulators are responding.',
    ]


def test_a_class_word_only_counts_when_whole():
    from src.modules.document_processor import clean_web_page

    _, text = clean_web_page(BeautifulSoup(
        '<body><div class="navigator-guide"><p>Kept</p></div><div class="share-bar">Share</div></body>', 'html.parser'
    ))

    assert text == 'Kept'


def test_an_article_header_stays_but_the_page_header_goes():
    from src.modules.document_processor import clean_web_page

    _, text = clean_web_page(BeautifulSoup(
        '<body><header>Site menu</header><article><header><h1>Headline</h1></header><p>Body</p></article></body>',
        'html.parser',
    ))

    assert text == 'Headline\nBody'


def test_adding_a_web_page_stores_only_its_clean_text(client, monkeypatch):
    import main
    scraped = BeautifulSoup(PAGE, 'html.parser')
    monkeypatch.setattr('src.modules.file_manager.WebBaseLoader', lambda url: MagicMock(scrape=lambda: scraped))
    import os
    os.makedirs('uploads', exist_ok=True)

    result = main.file_manager.webFileUpload('https://example.com/trust')

    assert result['success'] is True
    texts = [d[1] for d in _store()._docs if d[2].get('type') == 'web']
    stored = '\n'.join(texts)
    assert 'The danger in 2026' in stored
    assert 'All articles' not in stored and 'Download article' not in stored and 'cookies' not in stored
    assert [d[2]['title'] for d in _store()._docs if d[2].get('type') == 'web'][0] == 'The trust inversion | Institute'


def test_a_page_with_no_readable_text_is_refused(client, monkeypatch):
    import main
    empty = BeautifulSoup('<html><body><nav>Menu</nav><script>x()</script></body></html>', 'html.parser')
    monkeypatch.setattr('src.modules.file_manager.WebBaseLoader', lambda url: MagicMock(scrape=lambda: empty))

    result = main.file_manager.webFileUpload('https://example.com/empty')

    assert result == {'error': 'No readable text found on that page', 'success': False}
