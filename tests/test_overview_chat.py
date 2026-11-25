import io
import threading

from tests.conftest import FakeOllamaLLM

OVERVIEW = 'Overview of the notes.'
SEARCH_ANSWER = 'This is a fake LLM response.'


def _setup(client, *names):
    """Uploads text files (their overviews written by a separate fake model) and returns their ids"""
    import main
    main.overviews.llm = FakeOllamaLLM(canned_response=OVERVIEW)
    for name in names:
        client.post('/upload/file', data={'file': (io.BytesIO(f'Notes in {name}.\n'.encode()), name)},
                    content_type='multipart/form-data')
    by_name = {s['name']: s['id'] for s in client.get('/sources').get_json()['sources']}
    return [by_name[n] for n in names]


def _ask(client, question, session_id='s1', source_ids=None):
    body = {'question': question, 'session_id': session_id}
    if source_ids is not None:
        body['source_ids'] = source_ids
    return client.post('/chat', json=body).get_json()


def _spy_on_the_model(monkeypatch):
    """Counts calls to the chat model (the overviews were already written by their own fake)"""
    calls = []
    monkeypatch.setattr(FakeOllamaLLM, '_call', lambda self, prompt, *a, **kw: calls.append(prompt) or SEARCH_ANSWER)
    return calls


def test_summarize_answers_from_the_overview_without_the_model(client, monkeypatch):
    [notes] = _setup(client, 'notes.txt')
    model_calls = _spy_on_the_model(monkeypatch)

    result = _ask(client, 'Summarize this', source_ids=[notes])

    assert result['success'] is True
    assert result['answer'] == OVERVIEW
    assert model_calls == []
    assert [s['metadata']['source'] for s in result['sources']] == ['notes.txt']


def test_the_overview_answer_is_kept_in_the_chat(client):
    [notes] = _setup(client, 'notes.txt')

    _ask(client, 'Summarize this', source_ids=[notes])

    history = client.get('/chat/history/s1').get_json()
    assert [(h['question'], h['answer']) for h in history['history']] == [('Summarize this', OVERVIEW)]


def test_several_sources_get_one_overview_each_under_its_name(client):
    ids = _setup(client, 'a.txt', 'b.txt')

    answer = _ask(client, 'What are the main points?', source_ids=ids)['answer']

    assert answer == f'### a.txt\n\n{OVERVIEW}\n\n### b.txt\n\n{OVERVIEW}'


def test_more_than_five_sources_show_the_first_five(client):
    ids = _setup(client, *[f'notes{i}.txt' for i in range(7)])

    answer = _ask(client, 'Summarize these', source_ids=ids)['answer']

    assert answer.count('### ') == 5
    assert answer.endswith("_2 more sources aren't shown._")


def test_a_named_source_is_the_one_summarized(client):
    ids = _setup(client, 'alpha_notes.txt', 'beta_notes.txt')

    result = _ask(client, 'Summarize beta notes', source_ids=ids)

    assert result['answer'] == OVERVIEW
    assert [s['metadata']['source'] for s in result['sources']] == ['beta_notes.txt']


def test_chat_without_its_own_sources_searches(client):
    _setup(client, 'notes.txt')

    assert _ask(client, 'Summarize this')['answer'] == SEARCH_ANSWER


def test_a_specific_question_searches(client):
    [notes] = _setup(client, 'notes.txt')

    assert _ask(client, 'Summarize what it says about agents', source_ids=[notes])['answer'] == SEARCH_ANSWER


def test_a_source_without_an_overview_gets_one_when_asked(client):
    import main
    [notes] = _setup(client, 'notes.txt')
    main.overviews.delete(notes)

    assert _ask(client, 'Summarize this', source_ids=[notes])['answer'] == OVERVIEW


def test_an_overview_still_building_is_waited_for(client):
    import main
    [notes] = _setup(client, 'notes.txt')
    release = threading.Event()
    seen = {}

    class Slow(FakeOllamaLLM):
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            release.wait(5)
            return OVERVIEW

    main.overviews.inline = False
    main.overviews.llm = Slow()
    main.overviews.queue('notes.txt')

    def watch():
        while not release.is_set():
            progress = client.get('/chat/progress/s1').get_json()
            if progress['stage']:
                seen.update(progress)
                release.set()

    watcher = threading.Thread(target=watch)
    watcher.start()
    answer = _ask(client, 'Summarize this', source_ids=[notes])['answer']
    watcher.join(5)

    assert answer == OVERVIEW
    assert seen['stage'] == 'summarizing' and seen['source'] == 'notes.txt' and seen['since']
    assert client.get('/chat/progress/s1').get_json() == {'success': True, 'stage': None, 'source': None, 'since': None}


def test_no_overview_in_time_falls_back_to_search_with_a_note(client, monkeypatch):
    import main
    [notes] = _setup(client, 'notes.txt')
    release = threading.Event()

    class Stuck(FakeOllamaLLM):
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            release.wait(5)
            return OVERVIEW

    monkeypatch.setattr('src.config.OVERVIEW_WAIT_SECONDS', 0.2)
    main.overviews.inline = False
    main.overviews.llm = Stuck()
    main.overviews.queue('notes.txt')

    answer = _ask(client, 'Summarize this', source_ids=[notes])['answer']
    release.set()

    assert answer.startswith(SEARCH_ANSWER)
    assert "The full overview isn't ready yet" in answer


def test_a_failed_overview_falls_back_to_search_with_a_note(client):
    import main
    [notes] = _setup(client, 'notes.txt')

    class Broken(FakeOllamaLLM):
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            raise RuntimeError('Ollama is unreachable')

    main.overviews.llm = Broken()
    main.overviews.queue('notes.txt')

    answer = _ask(client, 'Summarize this', source_ids=[notes])['answer']

    assert answer.startswith(SEARCH_ANSWER) and "The full overview isn't ready yet" in answer


def test_ask_lens_is_unchanged(client, fake_review_client):
    fake_review_client.review.return_value = {'success': True, 'findings': [], 'warnings': []}
    review = client.post('/review', data={'file': (io.BytesIO(b'x = 1\n'), 'a.py')},
                         content_type='multipart/form-data').get_json()

    result = client.post('/chat', json={'question': 'Summarize this', 'session_id': review['session_id']}).get_json()

    assert result['answer'] == SEARCH_ANSWER


def test_with_overviews_switched_off_summarize_is_a_normal_search(client):
    import main
    [notes] = _setup(client, 'notes.txt')
    main.overviews.enabled = False
    main.overviews.delete(notes)

    answer = _ask(client, 'Summarize this', source_ids=[notes])['answer']

    assert answer == SEARCH_ANSWER  # no "isn't ready yet" note: it never will be
