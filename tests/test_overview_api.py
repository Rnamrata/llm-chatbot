import io

from langchain_core.documents import Document

NOTES = b'Gradient descent lowers the loss step by step.\n\nLearning rates control the step size.\n'


def _upload(client, name='notes.txt', content=NOTES):
    return client.post('/upload/file', data={'file': (io.BytesIO(content), name)}, content_type='multipart/form-data')


def _item(client, name):
    return next(s for s in client.get('/sources').get_json()['sources'] if s['name'] == name)


def _count_overview_calls(monkeypatch):
    import main
    calls = []
    real = main.overviews._build
    monkeypatch.setattr(main.overviews, '_build', lambda source: calls.append(source) or real(source))
    return calls


def test_an_added_file_gets_an_overview(client):
    _upload(client)

    item = _item(client, 'notes.txt')
    assert item['overview'] == 'ready'
    body = client.get(f"/sources/{item['id']}/overview").get_json()
    assert body['success'] is True and body['status'] == 'ready'
    assert body['text'] == 'This is a fake LLM response.' and body['built_at']


def test_a_reused_file_keeps_its_overview(client, monkeypatch):
    calls = _count_overview_calls(monkeypatch)

    _upload(client)
    _upload(client)

    assert calls == ['notes.txt']


def test_a_source_that_failed_to_upload_gets_no_overview(client, monkeypatch):
    calls = _count_overview_calls(monkeypatch)

    _upload(client, 'empty.txt', b'')

    assert calls == []


def test_deleting_a_source_deletes_its_overview(client):
    _upload(client)
    sid = _item(client, 'notes.txt')['id']

    client.delete(f'/sources/{sid}')

    assert client.get(f'/sources/{sid}/overview').status_code == 404


def test_an_older_source_can_be_summarized_on_request(client):
    import main
    main.vector_store.vectorstore.add_documents([Document(page_content='Old notes about tokenizers.',
                                                          metadata={'source': 'old.txt', 'type': 'file_upload', 'start_line': 1})])
    sid = _item(client, 'old.txt')['id']
    assert _item(client, 'old.txt')['overview'] is None

    response = client.post(f'/sources/{sid}/overview')

    assert response.status_code == 202
    assert response.get_json()['success'] is True
    assert client.get(f'/sources/{sid}/overview').get_json()['status'] == 'ready'


def test_a_source_without_an_overview_says_so(client):
    import main
    main.vector_store.vectorstore.add_documents([Document(page_content='x', metadata={'source': 'old.txt', 'type': 'file_upload'})])
    sid = _item(client, 'old.txt')['id']

    body = client.get(f'/sources/{sid}/overview').get_json()

    assert body['status'] is None and body['text'] is None


def test_unknown_and_legacy_sources_are_refused(client):
    import main
    main.vector_store.vectorstore.add_documents([Document(page_content='untyped', metadata={})])

    assert client.get('/sources/nope/overview').status_code == 404
    assert client.post('/sources/nope/overview').status_code == 404
    assert client.post('/sources/legacy/overview').status_code == 400


def test_with_overviews_switched_off_sources_carry_no_overview_field(client):
    import main
    main.overviews.enabled = False
    _upload(client)

    assert 'overview' not in _item(client, 'notes.txt')
