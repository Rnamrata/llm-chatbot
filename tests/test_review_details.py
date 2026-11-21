import io
from unittest.mock import MagicMock, patch

from src.modules.review_client import ReviewClient

JAVA = b'public class UserDao {\n  String q = "SELECT * FROM users WHERE name = \'" + name + "\'";\n}\n'

FINDINGS = [
    {'file': 'UserDao.java', 'line': 2, 'severity': 'high', 'message': 'SQL injection',
     'category': 'security', 'source': 'llm', 'suggestion': 'Use a PreparedStatement.'},
    {'file': 'UserDao.java', 'line': 1, 'severity': 'low', 'message': 'Missing Javadoc'},
]


def _post(client, name='UserDao.java', content=JAVA):
    return client.post(
        '/review',
        data={'file': (io.BytesIO(content), name)},
        content_type='multipart/form-data',
    )


def _stored_review_sources():
    import main
    return main.vector_store.vectorstore.get(where={'type': 'code_review_source'})['ids']


# ---------- GET /review/<id> returns everything POST returned ----------

def test_reload_keeps_category_source_and_suggestion(client, fake_review_client):
    fake_review_client.review.return_value = {'success': True, 'findings': FINDINGS, 'warnings': []}
    posted = _post(client).get_json()

    reloaded = client.get(f"/review/{posted['review_id']}").get_json()

    assert reloaded['findings'][0] == FINDINGS[0]
    assert reloaded['findings'][1] == FINDINGS[1]  # optional fields left out, not ''
    assert reloaded['session_id'] == posted['session_id']
    assert reloaded['filename'] == 'UserDao.java'
    assert reloaded['summary'] == posted['summary']


def test_reload_sorts_findings_most_serious_first(client, fake_review_client):
    fake_review_client.review.return_value = {'success': True, 'findings': list(reversed(FINDINGS)), 'warnings': []}
    posted = _post(client).get_json()

    reloaded = client.get(f"/review/{posted['review_id']}").get_json()

    assert [f['severity'] for f in reloaded['findings']] == ['high', 'low']


def test_reload_keeps_the_warnings_of_a_partial_review(client, fake_review_client):
    warnings = [{'source': 'llm', 'message': 'Ollama is unreachable'}]
    fake_review_client.review.return_value = {'success': True, 'findings': [], 'warnings': warnings}
    posted = _post(client).get_json()

    reloaded = client.get(f"/review/{posted['review_id']}").get_json()

    assert reloaded['warnings'] == warnings
    assert 'warning' not in reloaded


def test_reload_of_a_review_the_service_never_ran_keeps_its_warning(client, fake_review_client):
    fake_review_client.review.return_value = {'success': False, 'error': 'connection refused'}
    posted = _post(client).get_json()

    reloaded = client.get(f"/review/{posted['review_id']}").get_json()

    assert 'connection refused' in reloaded['warning']
    assert 'could not be reviewed' in reloaded['summary']


def test_post_response_names_the_file(client, fake_review_client):
    fake_review_client.review.return_value = {'success': True, 'findings': [], 'warnings': []}

    assert _post(client).get_json()['filename'] == 'UserDao.java'


# ---------- refused reviews are passed on, not stored ----------

def test_rate_limit_is_passed_on_with_retry_after(client, fake_review_client):
    fake_review_client.review.return_value = {
        'success': False, 'error': 'Too many requests', 'status': 429, 'retry_after': 270,
    }

    response = _post(client)

    assert response.status_code == 429
    assert response.headers['Retry-After'] == '270'
    assert response.get_json()['retry_after'] == 270
    assert _stored_review_sources() == []


def test_file_the_service_refuses_is_a_400_with_its_message(client, fake_review_client):
    fake_review_client.review.return_value = {
        'success': False, 'error': 'Unsupported file extension: .go', 'status': 400, 'retry_after': None,
    }

    response = _post(client)

    assert response.status_code == 400
    assert response.get_json()['error'] == 'Unsupported file extension: .go'
    assert _stored_review_sources() == []


def test_file_too_large_for_the_service_is_a_413(client, fake_review_client):
    fake_review_client.review.return_value = {
        'success': False, 'error': 'File too large', 'status': 413, 'retry_after': None,
    }

    assert _post(client).status_code == 413


def test_service_errors_other_than_refusals_still_store_the_code(client, fake_review_client):
    fake_review_client.review.return_value = {
        'success': False, 'error': 'Internal error', 'status': 500, 'retry_after': None,
    }

    response = _post(client)

    assert response.status_code == 200
    assert 'warning' in response.get_json()
    assert len(_stored_review_sources()) == 1


# ---------- /chat restores a lost review session ----------

def test_chat_with_review_id_restores_review_mode_for_an_unknown_session(client, fake_review_client):
    fake_review_client.review.return_value = {'success': True, 'findings': FINDINGS, 'warnings': []}
    review_id = _post(client).get_json()['review_id']

    body = client.post('/chat', json={
        'question': 'How do I fix line 2?', 'session_id': 'after-a-restart', 'review_id': review_id,
    }).get_json()

    assert body['success'] is True
    assert body['review_id'] == review_id


def test_chat_with_an_unknown_review_id_is_a_normal_chat(client):
    body = client.post('/chat', json={'question': 'Hi', 'session_id': 's1', 'review_id': 'nope'}).get_json()

    assert body['success'] is True
    assert 'review_id' not in body


# ---------- ReviewClient keeps the status ----------

def _client_answer(status, body, headers=None):
    response = MagicMock(status_code=status, text=str(body), headers=headers or {})
    response.json.return_value = body
    with patch('src.modules.review_client.requests.request', return_value=response):
        return ReviewClient(base_url='http://fake:4000', timeout=5).review('a.py', 'python', 'x = 1')


def test_client_reports_429_with_retry_after_and_the_service_message():
    result = _client_answer(
        429, {'success': False, 'error': 'Too Many Requests', 'message': 'Too many reviews, try again later'},
        {'Retry-After': '90'},
    )

    assert result == {
        'success': False, 'error': 'Too many reviews, try again later', 'status': 429, 'retry_after': 90,
    }


def test_client_falls_back_to_the_raw_text_when_the_body_is_not_json():
    response = MagicMock(status_code=502, text='Bad gateway', headers={})
    response.json.side_effect = ValueError()
    with patch('src.modules.review_client.requests.request', return_value=response):
        result = ReviewClient(base_url='http://fake:4000', timeout=5).review('a.py', 'python', 'x = 1')

    assert result['status'] == 502
    assert result['retry_after'] is None
    assert 'Bad gateway' in result['error']


# ---------- what the model reads ----------

def test_stored_findings_and_code_chunks_name_their_lines(client, fake_review_client):
    import main
    fake_review_client.review.return_value = {'success': True, 'findings': FINDINGS, 'warnings': []}
    review_id = _post(client).get_json()['review_id']

    stored = main.vector_store.vectorstore.get(where={'review_id': review_id})
    texts = dict(zip(stored['documents'], stored['metadatas']))
    finding_texts = [t for t, m in texts.items() if m['type'] == 'review_finding']
    code_texts = [t for t, m in texts.items() if m['type'] == 'code_review']

    assert 'UserDao.java, line 2 (high): SQL injection Suggestion: Use a PreparedStatement.' in finding_texts
    assert all(t.startswith('UserDao.java, lines ') for t in code_texts)