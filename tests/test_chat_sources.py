import io
from typing import Any, List

from langchain_core.documents import Document

from src.modules.chat_session import ChatSession
from src.modules.session_store import SessionStore
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


def _add(source, text, doc_type='file_upload', **extra):
    _store().add_documents([Document(page_content=text, metadata={'source': source, 'type': doc_type, **extra})])


def _ids(client):
    return {s['source']: s['id'] for s in client.get('/sources').get_json()['sources']}


def _upload(client, name, content, session_id=None):
    data = {'file': (io.BytesIO(content), name)}
    if session_id:
        data['session_id'] = session_id
    return client.post('/upload/file', data=data, content_type='multipart/form-data')


def _restarted():
    import main
    from src import config
    return ChatSession(main.llm_manager, main.vector_store, SessionStore(config.SESSIONS_FILE))


# ---------- choosing a chat's sources ----------

def test_a_chat_can_be_given_its_sources_before_the_first_question(client):
    _add('notes.pdf', 'Gradient descent lowers the loss.')
    _add('book.pdf', 'Chapter one is about trust.')
    ids = _ids(client)

    response = client.put('/chat/session/c1/sources', json={'source_ids': [ids['notes.pdf']]})

    body = response.get_json()
    assert response.status_code == 200
    assert body['session']['source_ids'] == [ids['notes.pdf']]
    assert body['session']['source_count'] == 1
    assert body['sources'] == [{'id': ids['notes.pdf'], 'name': 'notes.pdf', 'type': 'file_upload'}]


def test_a_chat_with_its_own_sources_only_searches_those(client):
    import main
    llm = RecordingLLM(prompts=[])
    main.llm_manager.llm = llm
    _add('notes.pdf', 'Gradient descent lowers the loss.')
    _add('book.pdf', 'Chapter one is about trust.')
    client.put('/chat/session/c1/sources', json={'source_ids': [_ids(client)['book.pdf']]})

    result = client.post('/chat', json={'question': 'What does it say about trust?', 'session_id': 'c1'}).get_json()

    assert [s['metadata']['source'] for s in result['sources']] == ['book.pdf']
    assert 'Chapter one is about trust.' in llm.prompts[-1]
    assert 'Gradient descent' not in llm.prompts[-1]


def test_an_empty_list_means_every_source(client):
    _add('notes.pdf', 'Gradient descent lowers the loss.')
    _add('book.pdf', 'Chapter one is about trust.')
    client.put('/chat/session/c1/sources', json={'source_ids': [_ids(client)['book.pdf']]})
    client.put('/chat/session/c1/sources', json={'source_ids': []})

    result = client.post('/chat', json={'question': 'Anything?', 'session_id': 'c1'}).get_json()

    assert sorted(s['metadata']['source'] for s in result['sources']) == ['book.pdf', 'notes.pdf']


def test_sources_sent_with_a_question_are_set_before_it_is_answered(client):
    _add('notes.pdf', 'Gradient descent lowers the loss.')
    _add('book.pdf', 'Chapter one is about trust.')
    ids = _ids(client)

    result = client.post('/chat', json={
        'question': 'What is this about?', 'session_id': 'c1', 'source_ids': [ids['notes.pdf'], 'not-a-source']
    }).get_json()

    assert [s['metadata']['source'] for s in result['sources']] == ['notes.pdf']
    assert result['source_ids'] == [ids['notes.pdf']]  # the unknown id is left out


def test_changing_the_sources_keeps_the_conversation(client):
    import main
    _add('notes.pdf', 'Gradient descent lowers the loss.')
    _add('book.pdf', 'Chapter one is about trust.')
    ids = _ids(client)
    client.post('/chat', json={'question': 'First question', 'session_id': 'c1'})

    client.put('/chat/session/c1/sources', json={'source_ids': [ids['book.pdf']]})
    result = client.post('/chat', json={'question': 'Second question', 'session_id': 'c1'}).get_json()

    assert result['message_count'] == 2
    assert [s['metadata']['source'] for s in result['sources']] == ['book.pdf']
    memory = main.chat_manager.sessions['c1']['memory'].chat_memory.messages
    assert memory[0].content == 'First question'


def test_bad_source_ids_are_refused(client):
    for bad in ('notes.pdf', [1, 2], None):
        response = client.put('/chat/session/c1/sources', json={'source_ids': bad})
        assert response.status_code == 400

    response = client.post('/chat', json={'question': 'x', 'session_id': 'c1', 'source_ids': 'abc'})
    assert response.status_code == 400


def test_a_review_chat_cannot_be_given_sources(client, fake_review_client):
    fake_review_client.review.return_value = {'success': True, 'findings': [], 'warnings': []}
    review = client.post(
        '/review', data={'file': (io.BytesIO(b'x = 1\n'), 'a.py')}, content_type='multipart/form-data'
    ).get_json()

    response = client.put(f"/chat/session/{review['session_id']}/sources", json={'source_ids': []})

    assert response.status_code == 400


# ---------- saved with the chat ----------

def test_history_and_the_session_list_show_the_chat_sources(client):
    _add('notes.pdf', 'Gradient descent lowers the loss.')
    _add('book.pdf', 'Chapter one is about trust.')
    ids = _ids(client)
    client.post('/chat', json={'question': 'Hi', 'session_id': 'c1', 'source_ids': [ids['notes.pdf'], ids['book.pdf']]})

    listed = next(s for s in client.get('/chat/sessions').get_json()['sessions'] if s['session_id'] == 'c1')
    history = client.get('/chat/history/c1').get_json()

    assert listed['source_count'] == 2
    assert sorted(listed['source_ids']) == sorted(ids.values())
    assert history['source_ids'] == [ids['notes.pdf'], ids['book.pdf']]


def test_a_chat_keeps_its_sources_after_a_restart(client):
    _add('notes.pdf', 'Gradient descent lowers the loss.')
    _add('book.pdf', 'Chapter one is about trust.')
    client.put('/chat/session/c1/sources', json={'source_ids': [_ids(client)['book.pdf']]})  # no question yet

    after = _restarted()
    result = after.query('What is this about?', 'c1')

    assert after.sessions['c1']['sources'] == ['book.pdf']
    assert [s['metadata']['source'] for s in result['sources']] == ['book.pdf']


# ---------- adding a source from a chat ----------

def test_a_file_added_from_a_chat_joins_its_sources(client):
    response = _upload(client, 'notes.txt', b'Gradient descent lowers the loss.', session_id='c1')

    body = response.get_json()
    assert response.status_code == 200
    assert body['session_id'] == 'c1'
    history = client.get('/chat/history/c1').get_json()
    assert history['source_ids'] == [body['source']['id']]


def test_a_link_added_from_a_chat_joins_its_sources(client, monkeypatch):
    import main

    def fake_web_upload(url):
        _add(url, 'A page about trust.', doc_type='web', title='Trust')
        return {'success': True, 'chunks_created': 1}
    monkeypatch.setattr(main.file_manager, 'webFileUpload', fake_web_upload)

    body = client.post('/upload/web', json={'url': 'https://example.com/t', 'session_id': 'c1'}).get_json()

    assert client.get('/chat/history/c1').get_json()['source_ids'] == [body['source']['id']]


# ---------- the same source added twice ----------

def test_the_same_file_added_again_is_reused_not_stored_twice(client):
    first = _upload(client, 'notes.txt', b'Gradient descent lowers the loss.').get_json()
    chunks = len(_store()._docs)

    again = _upload(client, 'notes.txt', b'Gradient descent lowers the loss.', session_id='c1').get_json()

    assert again['reused'] is True
    assert again['chunks_created'] == 0
    assert again['source']['id'] == first['source']['id']
    assert len(_store()._docs) == chunks
    assert client.get('/chat/history/c1').get_json()['source_ids'] == [first['source']['id']]


def test_a_file_with_the_same_name_and_new_content_replaces_the_old_copy(client):
    _upload(client, 'notes.txt', b'The old version of the notes.')

    again = _upload(client, 'notes.txt', b'The new version of the notes.').get_json()

    texts = [text for _id, text, meta in _store()._docs if meta.get('source') == 'notes.txt']
    assert again.get('replaced') is True
    assert texts == ['The new version of the notes.']


def test_a_link_added_again_is_reused_without_downloading_it(client, monkeypatch):
    import main
    calls = []

    def fake_web_upload(url):
        calls.append(url)
        _add(url, 'A page about trust.', doc_type='web', title='Trust')
        return {'success': True, 'chunks_created': 1}
    monkeypatch.setattr(main.file_manager, 'webFileUpload', fake_web_upload)

    client.post('/upload/web', json={'url': 'https://example.com/t'})
    again = client.post('/upload/web', json={'url': 'https://example.com/t'}).get_json()

    assert calls == ['https://example.com/t']
    assert again['reused'] is True


def test_a_failed_new_copy_leaves_the_old_one_in_place(client, monkeypatch):
    import main
    _upload(client, 'notes.txt', b'The old version of the notes.')
    monkeypatch.setattr(main.file_manager, 'uploadFile', lambda file, content_hash=None: {'success': False, 'error': 'broken'})

    response = _upload(client, 'notes.txt', b'A new version that fails.')

    assert response.status_code == 400
    texts = [text for _id, text, meta in _store()._docs if meta.get('source') == 'notes.txt']
    assert texts == ['The old version of the notes.']


# ---------- which source chips an answer shows ----------

def test_an_answer_only_shows_the_sources_it_names():
    from src.modules.document_context import sources_named_in
    docs = [
        Document(page_content='a', metadata={'source': 'Prompt_Engineering_for_LLMs_The_Art_and_Science_by_John_Berryman.pdf', 'type': 'file_upload'}),
        Document(page_content='b', metadata={'source': 'https://y.t/1', 'type': 'youtube', 'filename': 'Why Programmers adopt Bad Ideas - Casey Muratori_transcription.txt'}),
        Document(page_content='c', metadata={'source': 'https://k.com', 'type': 'web', 'title': 'The trust inversion | The Kyndryl Institute'}),
    ]

    assert [d.page_content for d in sources_named_in('The video "Why Programmers adopt Bad Ideas" by Casey Muratori says...', docs)] == ['b']
    assert [d.page_content for d in sources_named_in('The trust inversion says...', docs)] == ['c']
    assert [d.page_content for d in sources_named_in('As Prompt Engineering for LLMs explains...', docs)] == ['a']


def test_an_answer_naming_only_the_author_or_the_kind_shows_that_source():
    from src.modules.document_context import sources_named_in
    docs = [
        Document(page_content='a', metadata={'source': 'Prompt_Engineering_for_LLMs.pdf', 'type': 'file_upload'}),
        Document(page_content='b', metadata={'source': 'https://y.t/1', 'type': 'youtube', 'filename': 'Why Programmers adopt Bad Ideas - Casey Muratori_transcription.txt'}),
        Document(page_content='c', metadata={'source': 'https://k.com', 'type': 'web', 'title': 'Trust'}),
    ]

    assert [d.page_content for d in sources_named_in('Casey Muratori says programmers...', docs)] == ['b']
    assert [d.page_content for d in sources_named_in('The video explains...', docs)] == ['b']
    assert [d.page_content for d in sources_named_in('The book covers prompts.', docs)] == ['a']
    assert [d.page_content for d in sources_named_in('The article argues...', docs)] == ['c']


def test_short_or_common_words_do_not_count_as_naming_a_source():
    from src.modules.document_context import sources_named_in
    docs = [
        Document(page_content='a', metadata={'source': 'AI.pdf', 'type': 'file_upload'}),
        Document(page_content='b', metadata={'source': 'https://m.com', 'type': 'web', 'title': 'Medium'}),
    ]

    assert sources_named_in('AI is a broad field.', docs) == docs  # "AI" is too short to count


def test_an_answer_that_names_no_source_shows_them_all():
    from src.modules.document_context import sources_named_in
    docs = [
        Document(page_content='a', metadata={'source': 'notes.pdf', 'type': 'file_upload'}),
        Document(page_content='b', metadata={'source': 'book.pdf', 'type': 'file_upload'}),
    ]

    assert sources_named_in('Gradient descent lowers the loss.', docs) == docs


def test_a_chat_with_several_sources_reads_from_each_of_them(client):
    import main
    for i in range(12):
        _add('book.pdf', f'Book passage {i}.')
    _add('talk.mp3', 'The only passage of the talk.', doc_type='youtube')
    _add('other.pdf', 'Not chosen.')
    ids = _ids(client)

    result = client.post('/chat', json={
        'question': 'What do they say about passages?', 'session_id': 'c1', 'source_ids': [ids['book.pdf'], ids['talk.mp3']]
    }).get_json()

    used = [s['metadata']['source'] for s in main.chat_manager.sessions['c1']['messages'][-1]['sources']]
    assert 'talk.mp3' in used  # the small source is read too
    assert 'other.pdf' not in used
    assert len(used) == 8


def test_answer_instructions_name_no_real_source():
    from src.modules.llm_manager import DOCS_PROMPT_TEMPLATE

    assert 'trust inversion' not in DOCS_PROMPT_TEMPLATE.lower()
