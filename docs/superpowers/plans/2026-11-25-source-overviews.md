# Source Overviews Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** "Summarize this" in a document chat answers instantly from an overview of each source (a few sentences plus 5–7 key topics), built in the background when the source is added.

**Architecture:** llm-chatbot gains pure overview functions (topic clustering with numpy, prompts), an `Overviews` service (JSON store + one background worker thread), API fields and routes, and an overview path in `ChatSession.query`. Lens waits for uploads before asking, shows progress from `/chat/progress`, and shows overviews in the Sources library.

**Tech Stack:** Python 3.11 / Flask / LangChain Ollama / Chroma / numpy / pytest (llm-chatbot); Angular 22 signals, Vitest (Lens).

**Spec:** `docs/superpowers/specs/2026-11-25-source-overviews-design.md` (llm-chatbot)

## Global Constraints

- Local only: the document model (`LLM_MODEL`, llama3.2) with `LLM_NUM_CTX`, temperature 0, seed 42. No new Python or npm packages.
- Settings and defaults: `OVERVIEW_ENABLED=true`, `OVERVIEW_WHOLE_CHARS=20000`, `OVERVIEW_BATCH_CHARS=20000`, `OVERVIEW_TOPICS=20`, `OVERVIEW_WAIT_SECONDS=150`.
- Groups: `k = min(OVERVIEW_TOPICS, max(5, round(sqrt(n))))`, fixed seed; picked chunks in reading order (`start_line`).
- Overview text: 2–4 sentences, then 5–7 key topics as a bullet list; only what the passages say; in the source's language; no real source named in any prompt.
- Storage `data/overviews.json` next to `SESSIONS_FILE`, written via temp file + rename. Statuses: `building`, `ready`, `failed`; missing = none yet. `building` left over at start becomes none.
- Overview answers: document chats only; no model call; at most 5 sources.
- Tests never call Ollama. The user runs git: each task ends by listing files to commit, not by committing.

## Review Focus

1. A source with fewer chunks than 5 (a YouTube short with 1–2 chunks) is summarized whole, never clustered — Task 1 test `test_tiny_source_is_used_whole`.
2. "Summarize what it says about agents" is answered by search, not an overview — Task 5 phrasing table.
3. An overview question in a chat with no sources of its own goes through search as today — Task 5 `test_chat_without_its_own_sources_searches`.
4. Deleting a source while its overview is being built leaves no overview behind — Task 3 `test_delete_while_building_drops_the_result`.
5. A question sent while one upload is in progress and another has failed is asked once the first finishes, not stuck — Task 7 test.

---

## llm-chatbot

### Task 1: Pick representative passages

**Files:**
- Create: `src/modules/source_overview.py`
- Test: `tests/test_source_overview.py`

**Interfaces:**
- Produces: `Chunk = dict` with keys `text: str`, `start_line: int`, `embedding: list[float]`;
  `topic_count(n: int, topics: int) -> int`;
  `pick_passages(chunks: list[Chunk], whole_chars: int, topics: int, seed: int = 42) -> list[str]` — all chunk texts in reading order when their total ≤ `whole_chars` (or n < 5), else one chunk per k-means group (the one nearest its centre), in reading order.

- [ ] **Step 1: Write the failing tests**

```python
def test_topic_count_follows_the_rule():
    assert topic_count(9, 20) == 5          # max(5, round(sqrt 9)=3)
    assert topic_count(100, 20) == 10
    assert topic_count(900, 20) == 20       # capped by OVERVIEW_TOPICS

def test_short_source_is_used_whole_in_reading_order():
    chunks = [chunk('second', 20), chunk('first', 1)]
    assert pick_passages(chunks, whole_chars=1000, topics=20) == ['first', 'second']

def test_tiny_source_is_used_whole():
    assert pick_passages([chunk('x' * 30000, 1)], whole_chars=20000, topics=20) == ['x' * 30000]

def test_long_source_gets_one_passage_per_topic_in_reading_order():
    # 3 well-separated groups of 10 chunks each (embeddings near [1,0], [0,1], [-1,0]); k forced to 5 → each group
    # contributes, every pick is from the source, and picks are sorted by start_line
    picked = pick_passages(three_groups(), whole_chars=100, topics=5)
    assert len(picked) == 5 and len(set(picked)) == 5
    assert {group_of(p) for p in picked} == {'a', 'b', 'c'}
    assert picked == sorted(picked, key=line_of)

def test_same_source_gives_the_same_picks():
    assert pick_passages(three_groups(), 100, 5) == pick_passages(three_groups(), 100, 5)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_source_overview.py -q` — expect ImportError.
- [ ] **Step 3: Implement** with numpy only: k-means++ init from `np.random.default_rng(seed)`, at most 30 iterations, an empty group is re-seeded from the farthest point; cosine distance on normalized embeddings.
- [ ] **Step 4: Run** the same command — expect all pass.
- [ ] **Step 5: Files to commit:** `src/modules/source_overview.py`, `tests/test_source_overview.py`.

### Task 2: Write an overview from passages

**Files:**
- Modify: `src/modules/source_overview.py`
- Test: `tests/test_source_overview.py`

**Interfaces:**
- Consumes: `pick_passages` (Task 1).
- Produces: `OVERVIEW_PROMPT: str`, `COMBINE_PROMPT: str`;
  `write_overview(llm, name: str, passages: list[str], batch_chars: int) -> str` — one `llm.invoke` when the passages fit `batch_chars`; otherwise one call per batch (passages kept whole, in order) and one combine call over the batch results. Returns the stripped model text.

- [ ] **Step 1: Write the failing tests** (use a recording fake LLM like `RecordingLLM` in `tests/test_review_context.py`)

```python
def test_passages_that_fit_are_summarized_in_one_call():
    llm = recording('An overview.\n\n- Topic')
    assert write_overview(llm, 'book.pdf', ['a', 'b'], batch_chars=1000) == 'An overview.\n\n- Topic'
    assert len(llm.prompts) == 1 and 'a' in llm.prompts[0] and 'b' in llm.prompts[0]

def test_long_passages_are_summarized_in_batches_then_combined():
    llm = recording('part')
    write_overview(llm, 'book.pdf', ['x' * 600, 'y' * 600, 'z' * 600], batch_chars=1000)
    assert len(llm.prompts) == 4                       # 3 batches + 1 combine
    assert llm.prompts[-1].count('part') == 3

def test_the_prompts_ask_for_topics_and_name_no_real_source():
    for prompt in (OVERVIEW_PROMPT, COMBINE_PROMPT):
        assert '5' in prompt and '7' in prompt and 'bullet' in prompt.lower()
        for name in ('OWASP', 'trust inversion', 'LangChain', 'Logitech'):
            assert name.lower() not in prompt.lower()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_source_overview.py -q` — expect the new tests to fail.
- [ ] **Step 3: Implement** the two prompt constants (`{name}` and `{passages}` placeholders; rules from Global Constraints) and `write_overview`.
- [ ] **Step 4: Run** — expect all pass.
- [ ] **Step 5: Files to commit:** same two files.

### Task 3: Overview store and background worker

**Files:**
- Create: `src/modules/overviews.py`
- Modify: `src/config.py` (the five `OVERVIEW_*` settings; `OVERVIEWS_FILE = SESSIONS_FILE.parent / 'overviews.json'`)
- Modify: `tests/conftest.py` (`FakeChroma` keeps the embedding of each added document from `FakeEmbeddings`, and `get(include=[..., 'embeddings'])` returns `'embeddings'`; `FakeEmbeddings.embed_documents` returns a vector derived from the text, e.g. 8 floats from a hash, so groups differ)
- Test: `tests/test_overviews.py`

**Interfaces:**
- Consumes: `pick_passages`, `write_overview`; `source_id()` from `source_library.py`.
- Produces: `class Overviews(vector_store, llm, path, enabled=True, inline=False)`:
  - `get(sid: str) -> dict | None` — `{status, text, built_at, error}`;
  - `status(sid) -> 'ready' | 'building' | 'failed' | None`;
  - `queue(source: str, front: bool = False) -> None` — marks `building` and builds on the worker (with `inline=True`, builds before returning; tests use this);
  - `wait(sid: str, timeout: float) -> dict | None` — returns when the entry is no longer `building` or the timeout passes;
  - `delete(sid: str) -> None`;
  - `start() -> None` — starts the daemon thread (no-op when inline or disabled).
  Chunks for a source come from `vector_store.vectorstore.get(where={'source': source}, include=['documents', 'metadatas', 'embeddings'])`, document types only.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_queued_source_gets_a_ready_overview(store_with_book):
    o = Overviews(store_with_book, recording('Overview.'), tmp / 'overviews.json', inline=True)
    o.queue('book.pdf')
    assert o.get(source_id('book.pdf')) | {} >= {'status': 'ready', 'text': 'Overview.'}

def test_a_failed_build_is_kept_with_its_error(...):          # llm raises → status 'failed', error text kept
def test_overviews_survive_a_restart_but_building_does_not(...):  # write building+ready, new Overviews(): ready kept, building → None
def test_delete_while_building_drops_the_result(...):        # llm blocks on an Event; delete; release → get() is None
def test_wait_returns_when_ready_or_after_the_timeout(...):   # worker mode: wait(sid, 5) gets 'ready'; a stuck build returns 'building' after 0.2 s
def test_disabled_does_nothing(...):                          # enabled=False: queue() leaves get() None
```

- [ ] **Step 2: Run** `python -m pytest tests/test_overviews.py -q` — expect ImportError.
- [ ] **Step 3: Implement** `Overviews` (a `queue.Queue` plus a `deque` for `front=True`, one lock around the JSON dict; save through temp file + `os.replace` as in `SessionStore.save`; a build result is dropped if the entry was deleted meanwhile).
- [ ] **Step 4: Run** the new file, then `python -m pytest -q` — expect all pass (existing tests unaffected by the fake changes).
- [ ] **Step 5: Files to commit:** `src/modules/overviews.py`, `src/config.py`, `tests/conftest.py`, `tests/test_overviews.py`.

### Task 4: Build on upload; API for overviews

**Files:**
- Modify: `main.py` (create `overviews = Overviews(vector_store, llm_manager.llm, config.OVERVIEWS_FILE, enabled=config.OVERVIEW_ENABLED)` and `overviews.start()` when not testing; `_track_upload` queues the source after a successful upload; `delete_source` calls `overviews.delete(sid)`; `list_sources` adds `overview` per item; new routes)
- Test: `tests/test_overview_api.py`

**Interfaces:**
- Consumes: `Overviews` (Task 3).
- Produces (HTTP):
  - `GET /sources` items gain `overview: 'ready' | 'building' | 'failed' | null`.
  - `GET /sources/<id>/overview` → 200 `{success, id, status, text, built_at, error}`; 404 for an unknown source.
  - `POST /sources/<id>/overview` → 202 `{success, id, status: 'building'}`; 404 unknown; 400 for `legacy`.

- [ ] **Step 1: Write the failing tests** (the `app` fixture sets `main.overviews` to an inline `Overviews` with a fake LLM returning `'Overview.'`)

```python
def test_an_added_file_gets_an_overview(client): ...          # upload .txt → GET /sources item overview == 'ready'; GET .../overview text == 'Overview.'
def test_a_reused_file_keeps_its_overview(client): ...        # upload twice → one build (fake LLM called once)
def test_deleting_a_source_deletes_its_overview(client): ...  # DELETE /sources/<id> → GET .../overview 404
def test_an_older_source_can_be_summarized_on_request(client): ...  # store chunks directly, POST → 202, then GET → ready
def test_unknown_and_legacy_sources_are_refused(client): ...  # 404 and 400
```

- [ ] **Step 2: Run** `python -m pytest tests/test_overview_api.py -q` — expect failures.
- [ ] **Step 3: Implement** the wiring and routes in `main.py`.
- [ ] **Step 4: Run** `python -m pytest -q` — expect all pass.
- [ ] **Step 5: Files to commit:** `main.py`, `tests/conftest.py` (fixture), `tests/test_overview_api.py`.

### Task 5: Answer overview questions in a chat; progress

**Files:**
- Create: `src/modules/overview_questions.py`
- Modify: `src/modules/chat_session.py` (`ChatSession(..., overviews=None)`; overview path at the start of `query` for `docs` chats; `progress(session_id) -> dict`), `main.py` (pass `overviews`; `GET /chat/progress/<session_id>`)
- Test: `tests/test_overview_questions.py`, `tests/test_overview_chat.py`

**Interfaces:**
- Consumes: `Overviews.get/status/queue/wait` (Task 3); `display_name`, `source_id` (`source_library.py`); `_name_forms` matching from `document_context.py`.
- Produces: `is_overview_question(question: str) -> bool`;
  `ChatSession.progress(session_id) -> {'stage': 'summarizing' | None, 'source': str | None, 'since': str | None}`;
  `GET /chat/progress/<session_id>` returns that dict with `success: true`.
  Overview answer: one source → its text; several → `### <name>` then its text, for each (at most 5, then `_N more sources aren't shown._`). Response shape as today (`answer`, `sources` = one entry per source with `metadata.source`, `num_sources`).

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize('q', ['Summarize this', 'summarise the book', 'Give me an overview', 'What is this about?',
    'What are the main points?', 'key takeaways', 'tl;dr', 'What is the gist of it?'])
def test_overview_questions(q): assert is_overview_question(q)

@pytest.mark.parametrize('q', ['Summarize what it says about agents', 'What does chapter 3 say about RAG?',
    'What is a transformer?', 'main points about pricing in the video', 'Who wrote this?'])
def test_specific_questions_are_not(q): assert not is_overview_question(q)

def test_summarize_answers_from_the_overview_without_the_model(...):   # chat with one source, ready overview → answer == text; LLM not called
def test_several_sources_get_one_overview_each_under_its_name(...):    # 2 sources → '### a.txt' and '### b.txt' blocks
def test_a_named_source_is_the_one_summarized(...):                   # 'summarize b.txt' in a chat with a and b → only b
def test_chat_without_its_own_sources_searches(...):                  # no own sources → normal chain answer
def test_an_overview_still_building_is_waited_for(...):               # worker mode, slow fake → answer is the overview; progress() said summarizing <name> meanwhile
def test_no_overview_in_time_falls_back_to_search_with_a_note(...):  # OVERVIEW_WAIT_SECONDS tiny → search answer + "The full overview isn't ready yet"
def test_ask_lens_is_unchanged(...):                                  # review chat + 'summarize this' → review chain
```

- [ ] **Step 2: Run** `python -m pytest tests/test_overview_questions.py tests/test_overview_chat.py -q` — expect failures.
- [ ] **Step 3: Implement** `is_overview_question` (an overview word list plus a "specific" check: `about|on|regarding|chapter|section|part` followed by more words, or a question that is more than 12 words) and the `query` path: a source with no overview is queued with `front=True`, a `failed` one is not retried here (search answer with the note); record progress while waiting; clear it after.
- [ ] **Step 4: Run** `python -m pytest -q` — expect all pass.
- [ ] **Step 5: Files to commit:** the two new modules' files, `chat_session.py`, `main.py`, the two test files.

### Task 6: llm-chatbot README

**Files:** Modify `README.md`

- [ ] **Step 1:** Document: how overviews work (topic groups, built on add, a minute for a book), the five settings, `overview` in `GET /sources`, the two `/sources/<id>/overview` routes, `/chat/progress`, overview questions in chats.
- [ ] **Step 2:** Run `python -m pytest -q` — still all pass.
- [ ] **Step 3: Files to commit:** `README.md`, the spec and this plan.

## Lens (chatbot-frontend)

### Task 7: A question waits for files still being added

**Files:**
- Modify: `src/app/features/chat/chat.service.ts` (`send` queues the question while any `uploads()` entry is `adding`; asks when none is), `src/app/features/chat/message-list.ts/.html` (a waiting question shows "Waiting for <names> to be added · reading…")
- Test: `src/app/features/chat/chat-sources.spec.ts`

**Interfaces:**
- Produces: `ChatMessage` gains `waitingFor?: string[]` (file names) on the pending answer.

- [ ] **Step 1: Write the failing tests**

```ts
it('asks a question sent while a file is being added once the file is in', ...)
// drop a.pdf, send 'Summarize this' → expectNone /chat; '.thinking' says 'Waiting for a.pdf to be added'; flush upload → /chat asked
it('does not hold a question back for an upload that failed', ...)
// a.pdf adding, b.pdf failed → send → after a.pdf finishes, /chat asked once
```

- [ ] **Step 2: Run** `npx ng test --watch=false` — expect the two to fail.
- [ ] **Step 3: Implement** in `ChatService` (an effect-free check after each upload settles; no polling).
- [ ] **Step 4: Run** — expect all pass.
- [ ] **Step 5: Files to commit:** the three files.

### Task 8: Progress from the server while an answer is pending

**Files:**
- Modify: `src/app/features/chat/chat.service.ts` (while a docs answer is pending, `GET /chat/progress/<session>` every 2 s; stop when it settles), `chat.models.ts` (`ChatProgress`), `message-list` (pending text "Summarizing <source> · m:ss" when `stage === 'summarizing'`)
- Test: `src/app/features/chat/chat-page.spec.ts`

**Interfaces:**
- Consumes: `GET /chat/progress/<session_id>` (Task 5).
- Produces: `ChatProgress { stage: 'summarizing' | null; source: string | null; since: string | null }`.

- [ ] **Step 1: Write the failing test:** pending answer + progress flush `{stage:'summarizing', source:'book.pdf', since}` → `.thinking` contains `Summarizing book.pdf · 0:`; after the `/chat` flush, no more progress requests (`http.expectNone`).
- [ ] **Step 2–4:** run, implement (`interval(2000)` with `takeUntil` the answer settling; a failed progress call is ignored), run.
- [ ] **Step 5: Files to commit:** the changed files.

### Task 9: Overviews in the Sources library

**Files:**
- Modify: `src/app/features/sources/sources.models.ts` (`LibrarySource.overview?: 'ready' | 'building' | 'failed' | null`; `SourceOverview`), `sources.service.ts` (`overview(id)`, `summarize(id)`), `sources-table.html/.ts/.css` (per-row: *Summary* toggles the overview under the row as markdown via `app-markdown-answer`; *Summarizing…*; *Couldn't summarize · Try again*; *Summarize*), `sources-page.ts` (refresh while any row is `building`, every 5 s)
- Test: `src/app/features/sources/sources-page.spec.ts`

**Interfaces:**
- Consumes: `GET/POST /sources/<id>/overview`, `overview` in `GET /sources` (Task 4).
- Produces: `SourceOverview { status: 'ready' | 'building' | 'failed' | null; text: string | null; built_at: string | null; error: string | null }`.

- [ ] **Step 1: Write the failing tests**

```ts
it('opens a ready overview under its row', ...)        // click 'Summary' → GET .../overview → text shown
it('summarizes a source that has none yet', ...)       // click 'Summarize' → POST → row says 'Summarizing…'
it('offers to try again when summarizing failed', ...) // overview 'failed' → 'Couldn't summarize' + 'Try again' posts
```

- [ ] **Step 2–4:** run, implement, run; `npx ng build` has no budget warnings.
- [ ] **Step 5: Files to commit:** the changed files.

### Task 10: Lens README and the plan artifact

- [ ] **Step 1:** Lens README: phase table row "10. Source overviews"; the Testing list; the llm-chatbot PR it needs.
- [ ] **Step 2:** Mark #9 done in the Lens Change Plan artifact once both PRs are merged.
