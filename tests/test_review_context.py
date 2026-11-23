import io
from typing import Any, List

from langchain_core.documents import Document

from tests.conftest import FakeOllamaLLM

PYTHON = (
    "def average_price(products):\n"
    "    total = sum(p.price for p in products)\n"
    "    return total / len(products)\n"
    "\n"
    "def page(items, n):\n"
    "    return items[n * 10:(n + 1) * 10]\n"
)
FINDINGS = [
    {'file': 'shop.py', 'line': 5, 'severity': 'low', 'message': 'Magic number 10', 'category': 'maintainability'},
    {'file': 'shop.py', 'line': 3, 'severity': 'high', 'message': 'Division by zero on an empty list',
     'category': 'bug', 'source': 'llm', 'suggestion': 'Return 0 when products is empty'},
]


class RecordingLLM(FakeOllamaLLM):
    """Keeps every prompt it was sent"""
    prompts: List[str] = []

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        return self.canned_response


def _record(attr):
    import main
    llm = RecordingLLM(prompts=[])
    setattr(main.llm_manager, attr, llm)
    return llm


def _review(client, fake_review_client, code=PYTHON):
    fake_review_client.review.return_value = {'success': True, 'findings': FINDINGS, 'warnings': []}
    return client.post(
        '/review', data={'file': (io.BytesIO(code.encode()), 'shop.py')}, content_type='multipart/form-data'
    ).get_json()


def _ask(client, review, question):
    return client.post('/chat', json={'question': question, 'session_id': review['session_id']}).get_json()


# ---------- numbering ----------

def test_every_line_gets_its_number():
    from src.modules.review_context import number_lines

    assert number_lines('a\n  b\n') == '1: a\n2:   b'
    assert number_lines('x\ny', start=9).split('\n') == [' 9: x', '10: y']


def test_findings_are_listed_most_serious_first_with_their_fix():
    from src.modules.review_context import findings_text

    rows = findings_text([{**f, 'finding_source': f.get('source', '')} for f in FINDINGS]).split('\n')

    assert rows[0] == ('- line 3 (high, bug, llm): Division by zero on an empty list '
                       'Suggested fix: Return 0 when products is empty')
    assert rows[1] == '- line 5 (low, maintainability): Magic number 10'


# ---------- what Ask Lens reads ----------

def test_ask_lens_reads_the_whole_numbered_file_and_every_finding(client, fake_review_client):
    llm = _record('review_llm')
    review = _review(client, fake_review_client)
    import main
    main.vector_store.vectorstore.add_documents([
        Document(page_content='Unrelated notes', metadata={'source': 'notes.pdf', 'type': 'file_upload'})
    ])

    result = _ask(client, review, 'What does page do?')

    assert result['success'] is True
    prompt = llm.prompts[-1]
    assert 'File: shop.py (language: python). The complete file' in prompt
    assert '3:     return total / len(products)' in prompt
    assert '6:     return items[n * 10:(n + 1) * 10]' in prompt
    assert '- line 3 (high, bug, llm): Division by zero on an empty list' in prompt
    assert '- line 5 (low, maintainability): Magic number 10' in prompt
    assert 'Unrelated notes' not in prompt
    assert prompt.rstrip().endswith('Current Question: What does page do?\n\nAnswer:')


def test_a_follow_up_is_asked_as_it_was_written(client, fake_review_client):
    llm = _record('review_llm')
    review = _review(client, fake_review_client)

    _ask(client, review, 'How do I fix line 3?')
    _ask(client, review, 'Show only the changes')

    assert len(llm.prompts) == 2  # one model call per question: no rewrite first
    assert 'Current Question: Show only the changes' in llm.prompts[1]
    assert 'Developer: How do I fix line 3?' in llm.prompts[1]


def test_a_file_too_long_to_send_whole_gives_numbered_passages(client, fake_review_client):
    from src.modules.review_context import ReviewContextRetriever
    review = _review(client, fake_review_client)
    import main

    retriever = ReviewContextRetriever(vector_store=main.vector_store, review_id=review['review_id'], max_code_chars=10)
    docs = retriever.invoke('average price')

    assert 'too long to show whole' in docs[0].page_content
    assert '3:     return total / len(products)' in '\n'.join(d.page_content for d in docs)
    assert docs[-1].metadata['type'] == 'review_findings'


def test_ask_lens_uses_the_review_model_and_a_larger_context():
    from src import config
    import src.modules.llm_manager as llm_manager

    manager = llm_manager.LLMManager()

    assert manager.review_llm.model == config.REVIEW_LLM_MODEL
    assert manager.review_llm.num_ctx == config.REVIEW_NUM_CTX
    assert manager.review_llm.temperature == config.REVIEW_LLM_TEMPERATURE
    assert manager.review_llm.seed == config.REVIEW_LLM_SEED
    assert manager.llm.model == config.LLM_MODEL


def test_ask_lens_answers_the_same_way_each_time_by_default(monkeypatch):
    monkeypatch.delenv('REVIEW_LLM_TEMPERATURE', raising=False)
    monkeypatch.delenv('REVIEW_LLM_SEED', raising=False)
    monkeypatch.setattr('dotenv.load_dotenv', lambda *a, **kw: None)
    import importlib
    from src import config
    config = importlib.reload(config)

    assert config.REVIEW_LLM_TEMPERATURE == 0
    assert config.REVIEW_LLM_SEED == 42


# ---------- document chats ----------

def test_a_document_chat_answers_the_follow_up_as_it_was_written(client):
    llm = _record('llm')
    import main
    main.chat_manager.sessions.clear()
    main.vector_store.vectorstore.add_documents([
        Document(page_content='Gradient descent lowers the loss.', metadata={'source': 'notes.pdf', 'type': 'file_upload'})
    ])

    client.post('/chat', json={'question': 'What is gradient descent?', 'session_id': 'd1'})
    client.post('/chat', json={'question': 'Only the key number', 'session_id': 'd1'})

    answer_prompt = llm.prompts[-1]
    assert 'Current Question: Only the key number' in answer_prompt
