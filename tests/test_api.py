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