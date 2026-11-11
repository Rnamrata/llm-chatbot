import io


def _code_file(filename, content=b"def add(a, b):\n    return a + b\n"):
    return (io.BytesIO(content), filename)


def test_review_happy_path(client, fake_review_client):
    fake_review_client.review.return_value = {
        'success': True,
        'findings': [{'file': 'sample.py', 'line': 1, 'severity': 'low', 'message': 'add a docstring'}]
    }

    response = client.post(
        '/review',
        data={'file': _code_file('sample.py')},
        content_type='multipart/form-data'
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body['success'] is True
    assert 'review_id' in body
    assert 'session_id' in body
    assert 'summary' in body
    assert body['findings'][0]['message'] == 'add a docstring'
    assert 'def add' in body['code']


def test_review_rejects_bad_extension(client):
    response = client.post(
        '/review',
        data={'file': _code_file('not_code.exe')},
        content_type='multipart/form-data'
    )

    assert response.status_code == 400
    assert response.get_json()['success'] is False


def test_review_sanitizes_unsafe_filename(client, tmp_path):
    response = client.post(
        '/review',
        data={'file': _code_file('../../etc/passwd.py')},
        content_type='multipart/form-data'
    )

    assert response.status_code == 200
    assert response.get_json()['success'] is True

    # werkzeug's secure_filename strips path separators/traversal, so this
    # must land inside uploads/ and never escape to an ../../etc path
    assert (tmp_path / 'uploads' / 'etc_passwd.py').exists()
    assert not (tmp_path / 'etc' / 'passwd.py').exists()


def test_review_when_node_is_down_still_responds(client, fake_review_client):
    fake_review_client.review.return_value = {'success': False, 'error': 'connection refused'}

    response = client.post(
        '/review',
        data={'file': _code_file('sample.py')},
        content_type='multipart/form-data'
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body['success'] is True
    assert body['findings'] == []
    assert 'warning' in body

def _review(client):
    response = client.post(
        '/review',
        data={'file': _code_file('sample.py')},
        content_type='multipart/form-data'
    )
    assert response.status_code == 200
    return response.get_json()


def test_review_headline_counts_the_findings_exactly(client, fake_review_client):
    fake_review_client.review.return_value = {
        'success': True,
        'findings': [
            {'file': 'sample.py', 'line': 9, 'severity': 'high', 'message': 'SQL injection'},
            {'file': 'sample.py', 'line': 4, 'severity': 'medium', 'message': 'Hardcoded password'}
        ],
        'warnings': []
    }

    summary = _review(client)['summary']

    assert summary.startswith('2 issues found: 1 high, 1 medium.')
    assert 'Most serious: SQL injection (line 9).' in summary


def test_review_headline_is_not_written_by_the_language_model(client, fake_review_client):
    fake_review_client.review.return_value = {
        'success': True,
        'findings': [{'file': 'sample.py', 'line': 1, 'severity': 'low', 'message': 'add a docstring'}],
        'warnings': []
    }

    # FakeOllamaLLM answers every prompt with this text; the headline must not contain it
    assert 'fake LLM response' not in _review(client)['summary']


def test_review_returns_the_service_warnings_and_names_them_in_the_headline(client, fake_review_client):
    fake_review_client.review.return_value = {
        'success': True,
        'findings': [{'file': 'sample.py', 'line': 1, 'severity': 'low', 'message': 'add a docstring'}],
        'warnings': [{'source': 'llm', 'message': 'Ollama is unreachable at http://localhost:11434'}]
    }

    body = _review(client)

    assert body['warnings'] == [{'source': 'llm', 'message': 'Ollama is unreachable at http://localhost:11434'}]
    assert 'The AI reviewer reported a problem: Ollama is unreachable' in body['summary']


def test_review_with_no_findings_but_a_warning_is_not_called_clean(client, fake_review_client):
    fake_review_client.review.return_value = {
        'success': True,
        'findings': [],
        'warnings': [{'source': 'llm', 'message': 'timed out'}]
    }

    summary = _review(client)['summary']

    assert 'looks clean' not in summary
    assert summary.startswith('No issues found by the checks that ran.')


def test_review_with_no_findings_and_no_warnings_is_clean(client, fake_review_client):
    fake_review_client.review.return_value = {'success': True, 'findings': [], 'warnings': []}

    body = _review(client)

    assert body['summary'] == 'No issues found - the code looks clean!'
    assert body['warnings'] == []


def test_review_when_the_service_is_down_does_not_claim_the_code_is_clean(client, fake_review_client):
    fake_review_client.review.return_value = {'success': False, 'error': 'connection refused'}

    body = _review(client)

    assert 'looks clean' not in body['summary']
    assert 'could not be reviewed' in body['summary']
    assert body['warnings'] == []
    assert 'warning' in body


def test_review_response_always_has_a_warnings_list(client, fake_review_client):
    fake_review_client.review.return_value = {'success': True, 'findings': []}

    assert _review(client)['warnings'] == []