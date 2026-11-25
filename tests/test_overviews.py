import json
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.documents import Document

from src.modules.overviews import Overviews
from src.modules.source_library import source_id
from tests.conftest import FakeChroma, FakeEmbeddings, FakeOllamaLLM

BOOK = source_id('book.pdf')


class ScriptedLLM(FakeOllamaLLM):
    """Answers canned_response; can fail, or wait until released"""
    calls: int = 0
    error: str = ''
    release: Any = None

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.release is not None:
            self.release.wait(5)
        if self.error:
            raise RuntimeError(self.error)
        return self.canned_response


def store_with(source='book.pdf', chunks=3, doc_type='file_upload'):
    chroma = FakeChroma(embedding_function=FakeEmbeddings())
    chroma.add_documents([
        Document(page_content=f'{source} passage {i}', metadata={'source': source, 'type': doc_type, 'start_line': i * 10})
        for i in range(chunks)
    ])
    return SimpleNamespace(vectorstore=chroma)


def overviews(tmp_path, llm=None, store=None, **kwargs):
    return Overviews(store or store_with(), llm or ScriptedLLM(canned_response='Overview.\n- A topic'), tmp_path / 'overviews.json', **kwargs)


def test_a_queued_source_gets_a_ready_overview(tmp_path):
    o = overviews(tmp_path, inline=True)

    o.queue('book.pdf')

    entry = o.get(BOOK)
    assert entry['status'] == 'ready' and entry['text'] == 'Overview.\n- A topic' and entry['built_at']
    assert o.status(BOOK) == 'ready'


def test_the_model_reads_the_source_passages_in_order(tmp_path):
    prompts = []

    class Recording(ScriptedLLM):
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            prompts.append(prompt)
            return 'Overview.'

    overviews(tmp_path, llm=Recording(canned_response=''), inline=True).queue('book.pdf')

    assert prompts[0].index('passage 0') < prompts[0].index('passage 1') < prompts[0].index('passage 2')


def test_a_failed_build_is_kept_with_its_error(tmp_path):
    o = overviews(tmp_path, llm=ScriptedLLM(canned_response='', error='Ollama is unreachable'), inline=True)

    o.queue('book.pdf')

    assert o.get(BOOK)['status'] == 'failed'
    assert 'Ollama is unreachable' in o.get(BOOK)['error']


def test_a_source_with_no_chunks_fails_with_a_reason(tmp_path):
    o = overviews(tmp_path, store=store_with(chunks=0), inline=True)

    o.queue('book.pdf')

    assert o.get(BOOK)['status'] == 'failed'


def test_overviews_survive_a_restart_but_building_does_not(tmp_path):
    (tmp_path / 'overviews.json').write_text(json.dumps({'version': 1, 'overviews': {
        'a': {'status': 'ready', 'text': 'Kept.', 'built_at': '2026-11-25T10:00:00+00:00', 'error': None},
        'b': {'status': 'building', 'text': None, 'built_at': None, 'error': None},
    }}))

    o = overviews(tmp_path)

    assert o.get('a')['text'] == 'Kept.'
    assert o.get('b') is None


def test_a_broken_file_starts_empty(tmp_path):
    (tmp_path / 'overviews.json').write_text('{not json')

    assert overviews(tmp_path).get(BOOK) is None


def test_delete_while_building_drops_the_result(tmp_path):
    llm = ScriptedLLM(canned_response='Overview.', release=threading.Event())
    o = overviews(tmp_path, llm=llm)
    o.start()

    o.queue('book.pdf')
    _until(lambda: llm.calls == 1)
    o.delete(BOOK)
    llm.release.set()
    time.sleep(0.2)

    assert o.get(BOOK) is None


def test_wait_returns_when_ready(tmp_path):
    o = overviews(tmp_path)
    o.start()

    o.queue('book.pdf')

    assert o.wait(BOOK, 5)['status'] == 'ready'


def test_wait_gives_up_after_the_timeout(tmp_path):
    llm = ScriptedLLM(canned_response='Overview.', release=threading.Event())
    o = overviews(tmp_path, llm=llm)
    o.start()
    o.queue('book.pdf')

    started = time.monotonic()
    assert o.wait(BOOK, 0.2)['status'] == 'building'
    assert time.monotonic() - started < 2
    llm.release.set()


def test_a_source_asked_for_goes_to_the_front_of_the_queue(tmp_path):
    store = store_with('first.pdf')
    store.vectorstore.add_documents([Document(page_content='x', metadata={'source': 'second.pdf', 'type': 'file_upload'}),
                                     Document(page_content='y', metadata={'source': 'urgent.pdf', 'type': 'file_upload'})])
    order = []

    class Ordered(ScriptedLLM):
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            order.append(next(n for n in ('first', 'second', 'urgent') if f'"{n}"' in prompt))
            if len(order) == 1:
                gate.wait(5)
            return 'Overview.\n- A topic'

    gate = threading.Event()
    o = overviews(tmp_path, llm=Ordered(canned_response=''), store=store)
    o.start()
    o.queue('first.pdf')
    _until(lambda: order)
    o.queue('second.pdf')
    o.queue('urgent.pdf', front=True)
    gate.set()
    _until(lambda: len(order) == 3)

    assert order == ['first', 'urgent', 'second']


def test_code_review_chunks_are_never_summarized(tmp_path):
    store = store_with('Dao.java', doc_type='code_review')
    o = overviews(tmp_path, store=store, inline=True)

    o.queue('Dao.java')

    assert o.get(source_id('Dao.java'))['status'] == 'failed'


def test_disabled_does_nothing(tmp_path):
    o = overviews(tmp_path, enabled=False, inline=True)

    o.queue('book.pdf')

    assert o.get(BOOK) is None


def _until(condition, timeout=5):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail('timed out')
        time.sleep(0.01)
