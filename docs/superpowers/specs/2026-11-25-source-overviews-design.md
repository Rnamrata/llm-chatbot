# Source overviews ("summarize this") — design

Item #9 of the Lens change plan. Repos: **llm-chatbot** (most of the work) and **chatbot-frontend (Lens)**.

## Why

"Summarize this book" or "what is this video about" is answered today from the 8 passages that best
match the question, so the answer covers a few pages, not the source. People want a short overview of
what a source covers, to get the gist and decide what to ask next. Asking about a specific topic stays
with normal questions.

## Goals

- "Summarize this", "what's this about", "main points" and the like get an overview of the whole
  source (or of each of the chat's sources), **instantly** once the overview exists.
- An overview covers the source's main topics: a few sentences, then 5–7 key topics. Readable in about
  a minute (roughly 150–250 words per source).
- The work happens when a source is added, in the background, not when someone asks.
- A question sent while a file is still being added waits for the file (today it doesn't).

## Non-goals

- Chapter-by-chapter or section-by-section summaries. Too long to read; people ask about a topic
  themselves.
- Detecting chapters in PDFs.
- Summarizing a whole chat or a code review.
- Cloud models: everything stays on the local Ollama.

## Anti-goals (a failure even if it works)

- Asking for a summary of an existing book makes the person wait 10–15 minutes.
- The overview invents things the source doesn't say.
- Ordinary questions get slower, or get answered with an overview when they asked something specific.

## Approaches considered

- **A. Built in the background when a source is added; chat uses it** — *chosen.* Asking is instant
  once built; the cost is a background job, storage and recognizing overview questions.
- B. Built the first time someone asks, then kept. Simpler, but the first ask on a book waits about a
  minute, the kind of wait to avoid.
- C. Built at upload, shown only in the Sources library (like NotebookLM's source guide). Least work,
  but chat would still answer "summarize this" from 8 passages, which is the problem.

How an overview is made, chosen in the conversation: **topic clusters, not the whole text.** Long
sources don't fit llama3.2's 8K-token context (a book is ~200K tokens; summarizing it all would take
40–60 calls, 10–20 minutes). Instead, group the source's chunks by topic using the embeddings already
stored, take one representative passage per group, and summarize those in one or two calls: about a
minute for a book. A source short enough to fit is summarized whole.

## Design

### 1. Building an overview (llm-chatbot)

- **New module** `src/modules/source_overview.py`:
  - **Short source** (all chunk text ≤ `OVERVIEW_WHOLE_CHARS`, default 20,000 characters): its chunks in
    reading order (`start_line`), summarized in one call.
  - **Long source:** fetch its document chunks with their embeddings (Chroma `get(where=…,
    include=['documents','metadatas','embeddings'])`), run k-means with numpy (no new dependency),
    `k = min(OVERVIEW_TOPICS (20), max(5, round(sqrt(n))))`, a fixed seed so the same source gives the
    same groups. From each group take the chunk nearest its centre, then put the picked chunks in reading
    order. If they exceed the context (`OVERVIEW_BATCH_CHARS`, default 20,000), summarize them in two or
    three batches, then combine the batch summaries in a final call.
  - **Model:** the document model (`LLM_MODEL`, llama3.2) with `LLM_NUM_CTX`, temperature 0 and a fixed
    seed, like Ask Lens.
  - **Prompt:** summarize only what the passages say; first 2–4 sentences on what the source is and its
    main argument, then 5–7 key topics as a bullet list, each a short line. No invented facts, no
    "the passages say". Written in the source's language. It names no real source as an example (an earlier example name leaked into
    answers).
  - **Result:** `{ "text": "<markdown>", "built_at": ..., "chunks": n, "picked": m, "model": ... }`.
- **Storage:** `data/overviews.json` (next to `sessions.json`), keyed by source id (`source_id()` in
  `source_library.py`), written atomically like the session store. Status per source: `building`,
  `ready`, `failed` (with the error); no entry means "none yet".
- **Background worker:** one thread with a queue, one overview at a time (Ollama runs one model at a
  time anyway). A finished upload (file, YouTube, web) queues its source. Re-adding the same content
  reuses the source and so its overview. Deleting a source deletes its overview. On restart, entries
  left in `building` become "none yet" (built again when needed).
- **Sources added before this:** not built at startup (that would hold up Ollama). They are built when
  someone asks for an overview, or from the Sources library (see Lens).
- **Settings:** `OVERVIEW_ENABLED` (default `true`), `OVERVIEW_WHOLE_CHARS`, `OVERVIEW_BATCH_CHARS`,
  `OVERVIEW_TOPICS`, `OVERVIEW_WAIT_SECONDS` (default 150: Lens gives up on an answer after 180 s, so
  the server must answer before that).

### 2. Using overviews in a chat (llm-chatbot)

- **Document chats only.** Ask Lens (code-review chats) is unchanged.
- **Recognizing an overview question** is done in code, not by the model: a short question matching
  words like *summarize, summary, overview, main points/ideas/topics, key takeaways, gist, tl;dr,
  what is this/it about*. A question that also asks about something specific ("summarize what it says
  about agents") is **not** an overview question; it goes through normal search. Tested with a table
  of phrasings both ways.
- **Which sources:**
  - The source(s) the question names (the same name matching as `sources_named_in`).
  - Otherwise the chat's own sources.
  - If the chat has no sources of its own (it answers from all), the question goes through normal
    search, as today. *(My call: summarizing every source in the library would be long and rarely
    wanted.)*
- **Answer:** the stored overview text, no model call. With several sources, one overview per source
  under its name (at most 5; more than that: the first 5 and a line saying how many were left out).
  The source chips under the answer are those sources. The answer is saved in the chat like any other.
- **Not ready yet:** queue it (at the front) if needed and wait in the request for up to
  `OVERVIEW_WAIT_SECONDS`. If it isn't ready by then, or building fails, answer through normal search
  and add one line saying the full overview isn't available yet.
- **Progress:** `GET /chat/progress/<session_id>` returns what a pending question is waiting on:
  `{ "stage": "summarizing", "source": "<name>", "since": "<iso>" }` or `{ "stage": null }`. Kept in
  memory per session; general enough for other stages later.
- **API:** `GET /sources` gains `overview: "ready" | "building" | "failed" | null` per source.
  `GET /sources/<id>/overview` returns `{ status, text, built_at, error }`. `POST /sources/<id>/overview`
  queues a build (or a rebuild) and returns 202.

### 3. Lens

- **A question waits for files still being added**, as it already does for links: the question shows
  at once, and its answer is asked when every upload in progress has finished (or failed). The pending
  answer says "Waiting for *<file>* to be added · reading…". A failed upload doesn't block the question.
- **Pending answer text** comes from `/chat/progress` while a question is pending (polled every 2 s):
  "Summarizing *<source>* · 0:40". Otherwise the usual "Searching your sources…".
- **Sources library:** each row gets its overview status: *Summary* (opens the overview below the row),
  *Summarizing…*, *Couldn't summarize · Try again*, or *Summarize* for a source without one (queues it).
- Overview answers are ordinary markdown answers; nothing else changes in the chat.

## Constraints

- Local only: llama3.2 for overviews, nomic-embed-text embeddings already stored, no new Python
  packages (numpy is installed), no cloud calls.
- One Flask process; the worker is a daemon thread. Ollama handles one request at a time, so an overview
  being built delays chat answers a little while it runs (about a minute per book).
- Tests must not call Ollama (the existing fakes cover the LLM and Chroma).

## Settled during the builder check

- The wait is capped below Lens's 180 s answer timeout (`OVERVIEW_WAIT_SECONDS` = 150). A book asked
  about right after it finished uploading needs about a minute, well inside that. The time spent
  waiting for the upload itself doesn't count: Lens sends the question only once the file is in.
- The test fake for Chroma (`tests/conftest.py`) learns to return embeddings, so the clustering is
  tested without Ollama.
- All three source types have chunks with `start_line`, so reading order works for each.

## Left to the builder

- Exact prompt wording, k-means details (iterations, empty groups), JSON file locking.
- The exact word list for overview questions, within the rule above.
- Lens layout details of the overview in the Sources library, within the existing design.
- Docs: llm-chatbot README (settings, API, how overviews work), Lens README, plan artifact.
