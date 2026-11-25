from typing import Any, List

from src.modules.source_overview import pick_passages, topic_count
from tests.conftest import FakeOllamaLLM


def chunk(text, line, embedding=(1.0, 0.0)):
    return {'text': text, 'start_line': line, 'embedding': list(embedding)}


def three_groups():
    """30 chunks in three topics: a near [1,0], b near [0,1], c near [-1,0]; topics alternate through the source"""
    centres = {'a': (1.0, 0.0), 'b': (0.0, 1.0), 'c': (-1.0, 0.0)}
    chunks = []
    for i in range(30):
        group = 'abc'[i % 3]
        x, y = centres[group]
        wobble = (i % 7) * 0.01
        chunks.append(chunk(f'{group}:{i:02d}', i * 10 + 1, (x + wobble, y - wobble)))
    return chunks


def group_of(text):
    return text.split(':')[0]


def line_of(text):
    return int(text.split(':')[1])


# ---------- picking passages ----------

def test_topic_count_follows_the_rule():
    assert topic_count(9, 20) == 5
    assert topic_count(100, 20) == 10
    assert topic_count(900, 20) == 20


def test_short_source_is_used_whole_in_reading_order():
    chunks = [chunk('second', 20), chunk('first', 1)]
    assert pick_passages(chunks, whole_chars=1000, topics=20) == ['first', 'second']


def test_tiny_source_is_used_whole():
    assert pick_passages([chunk('x' * 30000, 1)], whole_chars=20000, topics=20) == ['x' * 30000]


def test_long_source_gets_one_passage_per_topic_in_reading_order():
    picked = pick_passages(three_groups(), whole_chars=100, topics=5)

    assert len(picked) == 5 and len(set(picked)) == 5
    assert {group_of(p) for p in picked} == {'a', 'b', 'c'}
    assert picked == sorted(picked, key=line_of)


def test_same_source_gives_the_same_picks():
    assert pick_passages(three_groups(), 100, 5) == pick_passages(three_groups(), 100, 5)


def test_chunks_without_a_line_keep_their_stored_order():
    chunks = [{'text': 'one', 'start_line': None, 'embedding': [1, 0]}, {'text': 'two', 'start_line': None, 'embedding': [0, 1]}]
    assert pick_passages(chunks, whole_chars=1000, topics=20) == ['one', 'two']


# ---------- writing the overview ----------

class RecordingLLM(FakeOllamaLLM):
    prompts: List[str] = []

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        return self.canned_response


def recording(answer):
    return RecordingLLM(canned_response=answer, prompts=[])


def test_passages_that_fit_are_summarized_in_one_call():
    from src.modules.source_overview import write_overview
    llm = recording('An overview.\n\n- Topic')

    assert write_overview(llm, 'book.pdf', ['alpha', 'beta'], batch_chars=1000) == 'An overview.\n\n- Topic'
    assert len(llm.prompts) == 1
    assert 'alpha' in llm.prompts[0] and 'beta' in llm.prompts[0] and '"book"' in llm.prompts[0]


def test_long_passages_are_summarized_in_batches_then_combined():
    from src.modules.source_overview import write_overview
    llm = recording('part\n- topic')

    write_overview(llm, 'book.pdf', ['x' * 600, 'y' * 600, 'z' * 600], batch_chars=1000)

    assert len(llm.prompts) == 4
    assert all(letter * 600 in llm.prompts[i] for i, letter in enumerate('xyz'))
    assert llm.prompts[-1].count('part') == 3


def test_the_prompts_ask_for_topics_and_name_no_real_source():
    from src.modules.source_overview import COMBINE_PROMPT, OVERVIEW_PROMPT
    from src.modules.source_overview import TOPICS_PROMPT
    for prompt in (OVERVIEW_PROMPT, COMBINE_PROMPT, TOPICS_PROMPT):
        assert '5' in prompt and '7' in prompt and '"- "' in prompt
        for name in ('OWASP', 'trust inversion', 'LangChain', 'Logitech'):
            assert name.lower() not in prompt.lower()


# ---------- tidying what the model wrote (found in a real run) ----------

def test_bullets_become_markdown_and_empty_ones_go():
    from src.modules.source_overview import tidy_overview
    text = 'A book about prompts.\n\nKey topics:\n• Context\n•\n* Retrieval\n· Evaluation\n•  \n'

    assert tidy_overview(text) == 'A book about prompts.\n\nKey topics:\n- Context\n- Retrieval\n- Evaluation'


def test_at_most_seven_topics_are_kept():
    from src.modules.source_overview import tidy_overview
    text = 'Intro.\n\n' + '\n'.join(f'- Topic {i}' for i in range(10))

    assert tidy_overview(text).count('\n- ') == 7


def test_topics_are_asked_for_again_when_none_came_back():
    from src.modules.source_overview import write_overview

    class TwoAnswers(RecordingLLM):
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            self.prompts.append(prompt)
            return 'An article about trust.\n\nKey topics:\n•\n•' if len(self.prompts) == 1 else '- Dependability\n- Accountability'

    llm = TwoAnswers(canned_response='', prompts=[])

    text = write_overview(llm, 'trust.html', ['passage'], batch_chars=1000)

    assert text == 'An article about trust.\n\nKey topics:\n- Dependability\n- Accountability'
    assert len(llm.prompts) == 2 and 'passage' in llm.prompts[1]


def test_the_model_gets_a_readable_name():
    from src.modules.source_overview import readable_name

    assert readable_name('LangChain_for_Life_Sciences_Ivan_Reznikov.pdf') == 'LangChain for Life Sciences Ivan Reznikov'
    assert readable_name('The trust inversion | The Kyndryl Institute') == 'The trust inversion | The Kyndryl Institute'
    assert readable_name('notes.txt') == 'notes'
