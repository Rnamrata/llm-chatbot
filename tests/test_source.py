import io

from langchain_core.documents import Document

NOTES = b"Supervised learning uses labelled examples. " * 40


def _upload_notes(client, name='notes.txt'):
    return client.post(
        '/upload/file',
        data={'file': (io.BytesIO(NOTES), name)},
        content_type='multipart/form-data',
    )


def _store():
    import main
    return main.vector_store.vectorstore


def test_upload_response_includes_the_stored_source(client):
    response = _upload_notes(client)

    assert response.status_code == 200
    source = response.get_json()['source']
    assert source['name'] == 'notes.txt'
    assert source['type'] == 'file_upload'
    assert source['status'] == 'ready'
    assert source['chunks'] == response.get_json()['chunks_created']
    assert source['added_at']


def test_list_groups_chunks_by_source(client):
    _upload_notes(client, 'a.txt')
    _upload_notes(client, 'b.txt')

    body = client.get('/sources').get_json()

    assert body['success'] is True
    assert sorted(s['name'] for s in body['sources']) == ['a.txt', 'b.txt']
    assert all(s['chunks'] > 0 for s in body['sources'])


def test_list_leaves_out_code_review_chunks(client):
    _store().add_documents([Document(page_content='class A {}', metadata={'source': 'A.java', 'type': 'code_review'})])

    assert client.get('/sources').get_json()['sources'] == []


def test_list_names_web_pages_by_title(client):
    _store().add_documents([Document(
        page_content='Python is a language.',
        metadata={'source': 'https://example.com/py', 'type': 'web', 'title': 'Python – Example'},
    )])

    [page] = client.get('/sources').get_json()['sources']
    assert page['name'] == 'Python – Example'
    assert page['type'] == 'web'


def test_chunks_without_a_type_are_one_outdated_item(client):
    _store().add_documents([
        Document(page_content='old', metadata={}),
        Document(page_content='old 2', metadata={'chunk_index': 3}),
        Document(page_content='old 3', metadata={'Header 1': 'Intro'}),
    ])

    [legacy] = client.get('/sources').get_json()['sources']
    assert legacy == {
        'id': 'legacy', 'name': 'Older data without details', 'type': 'unknown',
        'source': None, 'chunks': 3, 'status': 'outdated', 'added_at': None,
    }


def test_upload_in_progress_is_listed_as_processing(client):
    import main
    main.sources.start('https://youtu.be/abc', 'https://youtu.be/abc', 'youtube')

    [video] = client.get('/sources').get_json()['sources']
    assert video['status'] == 'processing'
    assert video['chunks'] == 0

    main.sources.finish('https://youtu.be/abc')
    assert client.get('/sources').get_json()['sources'] == []


def test_delete_removes_only_that_source(client):
    keep = _upload_notes(client, 'keep.txt').get_json()['source']
    drop = _upload_notes(client, 'drop.txt').get_json()['source']

    response = client.delete(f"/sources/{drop['id']}")

    assert response.status_code == 200
    assert response.get_json()['chunks_removed'] == drop['chunks']
    assert [s['id'] for s in client.get('/sources').get_json()['sources']] == [keep['id']]


def test_delete_legacy_removes_chunks_without_a_type(client):
    _upload_notes(client)
    _store().add_documents([Document(page_content='old', metadata={'chunk_index': 0})])

    assert client.delete('/sources/legacy').status_code == 200
    assert [s['name'] for s in client.get('/sources').get_json()['sources']] == ['notes.txt']


def test_delete_unknown_source_is_404(client):
    assert client.delete('/sources/doesnotexist').status_code == 404


def test_failed_upload_is_not_left_processing(client):
    response = client.post(
        '/upload/file',
        data={'file': (io.BytesIO(b'x'), 'image.png')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 400
    assert client.get('/sources').get_json()['sources'] == []