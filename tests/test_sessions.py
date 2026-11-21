import io
import json
from datetime import datetime, timedelta

from langchain_core.documents import Document

from src.modules.chat_session import ChatSession
from src.modules.session_store import SessionStore

JAVA = b'public class UserDao {\n  void find() {}\n}\n'
FINDINGS = [{'file': 'UserDao.java', 'line': 2, 'severity': 'high', 'message': 'SQL injection'}]


def _add_document(name='notes.pdf'):
    import main
    main.vector_store.vectorstore.add_documents([
        Document(page_content='Gradient descent lowers the loss.', metadata={'source': name, 'type': 'file_upload'})
    ])


def _review(client, fake_review_client, name='UserDao.java'):
    fake_review_client.review.return_value = {'success': True, 'findings': FINDINGS, 'warnings': []}
    return client.post(
        '/review', data={'file': (io.BytesIO(JAVA), name)}, content_type='multipart/form-data'
    ).get_json()


def _sessions(client):
    return client.get('/chat/sessions').get_json()['sessions']


def _restarted():
    """A second ChatSession reading the same file, as after restarting Flask"""
    import main
    from src import config
    return ChatSession(main.llm_manager, main.vector_store, SessionStore(config.SESSIONS_FILE))


# ---------- what History lists ----------

def test_a_chat_is_listed_with_a_title_preview_and_counts(client):
    _add_document()
    client.post('/chat', json={'question': 'What is gradient descent?', 'session_id': 's1'})
    client.post('/chat', json={'question': 'Give me an example', 'session_id': 's1'})

    [chat] = _sessions(client)

    assert chat['session_id'] == 's1'
    assert chat['type'] == 'docs'
    assert chat['title'] == 'What is gradient descent?'
    assert chat['preview'] == 'Give me an example'
    assert chat['message_count'] == 2
    assert chat['messages'] == 4
    assert chat['source_count'] == 1
    assert chat['pinned'] is False


def test_a_review_is_listed_by_file_name_with_its_summary(client, fake_review_client):
    review = _review(client, fake_review_client)

    [listed] = _sessions(client)

    assert listed['session_id'] == review['session_id']
    assert listed['type'] == 'code_review'
    assert listed['title'] == 'Review: UserDao.java'
    assert listed['preview'].startswith('1 issue found')
    assert listed['review_id'] == review['review_id']
    assert listed['language'] == 'java'
    assert listed['messages'] == 1


def test_an_empty_session_is_not_listed(client):
    client.post('/chat/new')

    assert _sessions(client) == []


def test_sessions_are_listed_most_recent_first(client, fake_review_client):
    client.post('/chat', json={'question': 'First', 'session_id': 'old'})
    client.post('/chat', json={'question': 'Second', 'session_id': 'new'})

    assert [s['session_id'] for s in _sessions(client)] == ['new', 'old']


def test_long_first_questions_are_shortened_for_the_title(client):
    client.post('/chat', json={'question': 'word ' * 40, 'session_id': 's1'})

    title = _sessions(client)[0]['title']
    assert len(title) <= 60
    assert title.endswith('…')


# ---------- pin and rename ----------

def test_pin_and_rename_a_session(client):
    client.post('/chat', json={'question': 'Hi', 'session_id': 's1'})

    pinned = client.patch('/chat/session/s1', json={'pinned': True}).get_json()
    renamed = client.patch('/chat/session/s1', json={'title': 'ML notes'}).get_json()

    assert pinned['session']['pinned'] is True
    assert renamed['session']['title'] == 'ML notes'
    assert _sessions(client)[0]['pinned'] is True


def test_pin_checks_its_input(client):
    client.post('/chat', json={'question': 'Hi', 'session_id': 's1'})

    assert client.patch('/chat/session/nope', json={'pinned': True}).status_code == 404
    assert client.patch('/chat/session/s1', json={'pinned': 'yes'}).status_code == 400
    assert client.patch('/chat/session/s1', json={'title': '  '}).status_code == 400


# ---------- history keeps sources ----------

def test_history_returns_each_answer_with_its_sources(client):
    _add_document()
    client.post('/chat', json={'question': 'What is gradient descent?', 'session_id': 's1'})

    history = client.get('/chat/history/s1').get_json()

    assert history['title'] == 'What is gradient descent?'
    assert history['type'] == 'docs'
    assert history['history'][0]['sources'][0]['metadata']['source'] == 'notes.pdf'


# ---------- saved across a restart ----------

def test_chats_survive_a_restart_with_their_sources_and_pins(client):
    _add_document()
    client.post('/chat', json={'question': 'What is gradient descent?', 'session_id': 's1'})
    client.patch('/chat/session/s1', json={'pinned': True})

    after = _restarted()

    [listed] = after.list_sessions()['sessions']
    assert listed['title'] == 'What is gradient descent?'
    assert listed['pinned'] is True
    history = after.get_history('s1')['history']
    assert history[0]['question'] == 'What is gradient descent?'
    assert history[0]['sources'][0]['metadata']['source'] == 'notes.pdf'


def test_a_restored_chat_can_carry_on(client):
    client.post('/chat', json={'question': 'First question', 'session_id': 's1'})

    after = _restarted()
    result = after.query('Second question', 's1')

    assert result['success'] is True
    assert result['message_count'] == 2
    memory = after.sessions['s1']['memory'].chat_memory.messages
    assert [m.content for m in memory][:1] == ['First question']  # earlier turns were replayed


def test_a_restored_review_stays_scoped_to_its_code(client, fake_review_client):
    review = _review(client, fake_review_client)
    _add_document()  # unrelated document that must not leak into the review chat

    after = _restarted()
    result = after.query('How do I fix line 2?', review['session_id'])

    assert result['review_id'] == review['review_id']
    assert after.sessions[review['session_id']]['chain'].retriever.filter == {'review_id': review['review_id']}


def test_an_unreadable_sessions_file_is_set_aside(app):
    from src import config
    config.SESSIONS_FILE.write_text('{not json', encoding='utf-8')

    after = _restarted()

    assert after.sessions == {}
    assert config.SESSIONS_FILE.with_suffix('.broken.json').exists()


# ---------- delete and clean up ----------

def test_deleting_a_review_session_removes_it_and_its_review(client, fake_review_client):
    import main
    from src import config
    review = _review(client, fake_review_client)

    client.delete(f"/chat/clear/{review['session_id']}")

    assert _sessions(client) == []
    assert main.vector_store.vectorstore.get(where={'review_id': review['review_id']})['ids'] == []
    assert json.loads(config.SESSIONS_FILE.read_text())['sessions'] == {}


def test_cleanup_removes_old_chats_but_keeps_pinned_ones(client):
    import main
    for sid in ('old', 'old-pinned', 'recent'):
        client.post('/chat', json={'question': f'Question {sid}', 'session_id': sid})
    client.patch('/chat/session/old-pinned', json={'pinned': True})
    long_ago = datetime.now() - timedelta(hours=30)
    for sid in ('old', 'old-pinned'):
        main.chat_manager.sessions[sid]['last_activity'] = long_ago

    result = client.post('/chat/cleanup', json={'inactive_hours': 24}).get_json()

    assert result['cleaned'] == 1
    assert sorted(s['session_id'] for s in _sessions(client)) == ['old-pinned', 'recent']