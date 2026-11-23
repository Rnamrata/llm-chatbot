# 🤖 RAG Chatbot System (Lens)

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot system that allows you to chat with your documents using local LLMs. Upload files, YouTube videos, or web pages, and have intelligent conversations with the content.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
[![CI](https://github.com/Rnamrata/llm-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Rnamrata/llm-chatbot/actions/workflows/ci.yml)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-orange.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Performance Tips](#-performance-tips)
- [Contributing](#-contributing)

---

## ✨ Features

### Core Capabilities
- 📄 **Multi-Source Document Upload** - PDF, TXT, Markdown, YouTube, Web pages
- 🧠 **Intelligent Conversation** - Context-aware with conversation history
- 🔍 **Semantic Search** - Vector-based retrieval with source attribution
- 🚀 **Automatic Pipeline** - One-call upload to storage
- 💾 **Saved Sessions** - Chats and code reviews are saved to disk with a title, type and pin, so they survive a restart

### Technical Features
- 🏗️ **Modular Architecture** - Clean separation of concerns
- 🔌 **RESTful API** - Easy frontend integration
- 🎯 **Local LLM** - Privacy-focused with Ollama
- 📊 **Monitoring** - Built-in stats and health checks
- 🛡️ **Error Handling** - Comprehensive error management

---

## 🏛️ Architecture
```
     ┌──────────────────────────────────┐
     │  Client (frontend / API caller)  │
     └──────────────────────────────────┘
                      │
                 REST (CORS)
                      ▼
     ┌──────────────────────────────────┐
     │           llm-chatbot            │
     │         Flask API :5001          │
     │    FileManager / ChatSession     │
     │  DocumentProcessor / LLMManager  │
     └──────────────────────────────────┘
                       │
           ┌───────────┴─────────────┐
           │                         │
┌────────────────────┐      ┌────────────────┐
│       Ollama       │      │    ChromaDB    │
│  LLM + embeddings  │      │  vector store  │
│       :11434       │      │  ./chroma_db   │
└────────────────────┘      └────────────────┘
```

**Data Flow:**
1. **Upload:** Document → Extract → Chunk → Embed → Store
2. **Chat:** Question → search the added sources only (never code reviews), 8 passages → each passage labelled with its source's name, kind and when it was added → LLM → Answer with Sources. Passages are embedded with nomic's `search_document: ` prefix and questions with `search_query: `
3. **Code Review:** Code file → validated & stored → sent to the configured review service → findings stored → opening message built from the findings (exact counts, plus a note if a check had a problem) → chat session scoped to that review via `review_id`
4. **Code Review chat (Ask Lens):** Question → the whole file with a number on every line, plus every finding → qwen2.5-coder:7b → Answer. A follow-up is sent as it was asked, not rewritten first. Only a file longer than `REVIEW_FULL_CODE_CHARS` is searched instead, and those passages are numbered too

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask 3.0+ |
| LLM | Ollama (llama3.2 for document chats, qwen2.5-coder:7b for code-review chats) |
| Embeddings | nomic-embed-text |
| Vector DB | ChromaDB |
| Framework | LangChain |
| PDF Processing | PyPDF2 |
| Audio | OpenAI Whisper |
| Video | yt-dlp |

---

## 📦 Installation

### Prerequisites

- Python 3.11+ (CI runs 3.12)
- Ollama
- FFmpeg (for YouTube support)

### Quick Install
```bash
# 1. Install Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Windows: Download from https://ollama.com/download

# 2. Start Ollama and pull models
ollama serve
ollama pull llama3.2
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text

# 3. Install FFmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt-get install ffmpeg

# 4. Clone and setup project
git clone https://github.com/Rnamrata/llm-chatbot.git
cd llm-chatbot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 5. Create directories
mkdir -p uploads/media chroma_db
```

### Environment Variables

Copy the example file and adjust as needed:
```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `FLASK_HOST` / `FLASK_PORT` / `FLASK_DEBUG` | `0.0.0.0` / `5001` / `false` | Flask server bind address |
| `CORS_ORIGINS` | `http://localhost:4200` | Comma-separated list of allowed frontend origins |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama is running |
| `LLM_MODEL` / `LLM_TEMPERATURE` | `llama3.2` / `0.7` | Chat model and temperature |
| `LLM_NUM_CTX` | `8192` | Tokens the chat model reads at once: the passages and the chat history |
| `REVIEW_LLM_MODEL` / `REVIEW_LLM_TEMPERATURE` / `REVIEW_LLM_SEED` | `qwen2.5-coder:7b` / `0` / `42` | Model, temperature and seed for code-review chats (Ask Lens); 0 and a fixed seed give the same answer to the same question |
| `REVIEW_NUM_CTX` | `16384` | Tokens the review model reads at once: the whole numbered file, the findings and recent messages. Keep it equal to the review service's `LLM_NUM_CTX`: both use the same model, and Ollama reloads it to change the context size |
| `REVIEW_FULL_CODE_CHARS` | `30000` | Longest file (in characters) sent whole; a longer one is searched instead |
| `REVIEW_HISTORY_MESSAGES` | `8` | Earlier messages included in a code-review chat prompt |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `EMBED_DOCUMENT_PREFIX` / `EMBED_QUERY_PREFIX` | `search_document: ` / `search_query: ` | Task prefixes nomic-embed-text expects; empty for a model without them. After changing either, run `scripts/reembed.py` |
| `CHROMA_PERSIST_DIR` / `CHROMA_COLLECTION` | `./chroma_db` / `my_documents` | Vector store location/collection |
| `RETRIEVAL_K` | `8` | Default number of chunks retrieved per query (document chats, and very long files in code-review chats) |
| `EMBED_BATCH_SIZE` / `EMBED_ATTEMPTS` | `64` / `3` | Chunks embedded per Ollama request, and tries per batch before an upload fails |
| `UPLOAD_DIR` / `MEDIA_DIR` | `./uploads` / `./uploads/media` | Where uploaded files/audio are saved |
| `SESSIONS_FILE` | `./data/sessions.json` | Where chat sessions are saved |
| `MAX_UPLOAD_MB` | `20` | Max upload size |
| `WHISPER_MODEL` | `base` | Whisper model for YouTube transcription; loaded on first use (the first transcription downloads it) |
| `REVIEW_SERVICE_URL` / `REVIEW_TIMEOUT_SECONDS` | `http://localhost:4000` / `120` | Where the code review service is running, and how long to wait for a review (longer than its `LLM_TIMEOUT_MS`, 90 s). The health check waits 5 s |

---

## 🚀 Quick Start

Start Ollama before the Flask server — the server depends on it being up:

```bash
# Terminal 1: Ollama (LLM + embeddings)
ollama serve

# Terminal 2: Flask server
cd llm-chatbot
source .venv/bin/activate
python main.py

# Terminal 3 (optional): manual smoke test against the running server
python scripts/smoke_test.py
```

Server will be available at: `http://localhost:5001`

### Re-embedding what is stored

After changing `EMBEDDING_MODEL`, `EMBED_DOCUMENT_PREFIX` or `EMBED_QUERY_PREFIX`, re-embed the chunks already stored, so old vectors match new questions. The text and metadata stay as they are. Stop Flask first and back up `chroma_db/`:

```bash
cp -R chroma_db chroma_db.backup
python scripts/reembed.py --dry-run   # count the chunks
python scripts/reembed.py             # about 1 minute per 2,000 chunks
```

Web pages added before the cleaning was introduced keep their menus and footers; remove and add them again to store only the article.

---

## 📡 API Reference

**Base URL:** `http://localhost:5001`

### Upload Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/upload/file` | POST | `file: <file>` | Upload PDF/TXT/MD |
| `/upload/youtube` | POST | `{"url": "..."}` | Transcribe YouTube video |
| `/upload/web` | POST | `{"url": "..."}` | Scrape a web page and store its readable text, without menus, footers, banners and "related articles" |
| `/upload/code` | POST | `file: <code file>`, `session_id` | Upload code for review without attaching it to a chat session |

### Code Review Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/review` | POST | `file: <code file>`, `session_id` (optional) | Review a code file, store the code + findings, and attach the review to a chat session (creates one if `session_id` is omitted). Returns `summary` (the opening message), `filename`, `findings`, `warnings`, `code` and `language`; `warning` is added only when the review service was unreachable. When the review service refuses the file, its status is passed on and nothing is stored: 400 (unsupported or bad input), 413 (too large) or 429 (too many reviews, with `Retry-After` and `retry_after` in the body) |
| `/review/{review_id}` | GET | - | Reload a stored review: the same fields `POST /review` returned, findings most serious first. 404 if it was deleted |

### Source Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/sources` | GET | - | List added files, YouTube videos and web pages, newest first: `id`, `name`, `type`, `source`, `chunks`, `status` (`ready`, `processing` or `outdated`) and `added_at`. Chunks stored by older versions without a type show as one `outdated` item with id `legacy` |
| `/sources/{id}` | DELETE | - | Remove a source's chunks from the vector store (the uploaded file stays on disk) |

### Chat Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/chat/new` | POST | - | Create new session |
| `/chat` | POST | `{"question": "...", "session_id": "...", "review_id": "..."}` | Send message. `review_id` is optional: for a review's chat the server no longer knows, it restores review mode |
| `/chat/history/{id}` | GET | - | Get conversation history, each answer with its sources |
| `/chat/session/{id}` | GET | - | Get session info |
| `/chat/session/{id}` | PATCH | `{"pinned": true}` or `{"title": "..."}` | Pin, unpin or rename a session |
| `/chat/clear/{id}` | DELETE | - | Delete a session; for a review's session, the stored review goes too |
| `/chat/sessions` | GET | - | List saved chats and reviews, most recently used first, with `type`, `title`, `preview`, `pinned` and counts |
| `/chat/cleanup` | POST | `{"inactive_hours": 24, "keep_pinned": true}` | Delete sessions not used for a while, and their reviews; pinned ones are kept unless `keep_pinned` is `false` |

Sessions are saved to `SESSIONS_FILE` (`data/sessions.json` by default) whenever they change, and loaded when the server starts. A session's conversation is rebuilt from its saved messages the first time it is used again.

### Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — also reports Ollama reachability and review-service status |
| `/stats` | GET | Database statistics — also includes total stored reviews |

### Example Responses

**Upload Success:**
```json
{
  "success": true,
  "chunks_created": 45,
  "filename": "document.pdf"
}
```

**Chat Response:**
```json
{
  "success": true,
  "answer": "Machine learning is...",
  "sources": [...],
  "num_sources": 5,
  "session_id": "abc-123"
}
```

**Review Response:**
```json
{
  "success": true,
  "session_id": "abc-123",
  "review_id": "def-456",
  "filename": "UserDao.java",
  "summary": "2 issues found: 1 high, 1 medium. Most serious: SQL injection vulnerability (line 9).\n\nNote: The AI reviewer reported a problem: Ollama is unreachable at http://localhost:11434.",
  "findings": [
    {
      "file": "UserDao.java",
      "line": 9,
      "severity": "high",
      "category": "security",
      "message": "SQL injection vulnerability",
      "source": "llm"
    }
  ],
  "warnings": [
    { "source": "llm", "message": "Ollama is unreachable at http://localhost:11434" }
  ],
  "code": "...",
  "language": "java"
}
```

The `summary` is built in code from the findings, so its counts always match the list. `warnings` are problems one part of the review had while the rest still completed. If the review service cannot be reached at all, `findings` is empty, `summary` says the code could not be reviewed, and a `warning` field explains why.

---

## 📁 Project Structure
```
llm-chatbot/
├── main.py                     # Flask app entry point
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
│
├── src/modules/
│   ├── file_manager.py         # Upload & processing
│   ├── document_processor.py   # Text extraction & chunking
│   ├── vector_store_and_embedding.py  # Vector operations
│   ├── llm_manager.py          # LLM management
│   ├── chat_session.py         # Session management
│   ├── session_store.py        # Saves sessions to data/sessions.json
│   ├── source_library.py       # Lists and removes sources
│   ├── review_summary.py       # Builds the review's opening message
│   ├── review_context.py       # What a code-review chat reads: numbered code + findings
│   ├── document_context.py     # What a document chat reads: passages labelled with their source
│   └── review_client.py        # Code review service client
│
├── src/config.py                # Central configuration (reads .env)
├── .env.example                 # Environment variable template
├── pytest.ini                   # pytest configuration
│
├── .github/workflows/
│   └── ci.yml                   # Runs the tests on every pull request
│
├── tests/                       # Automated tests (pytest, fully mocked)
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_document_processor.py
│   ├── test_document_search.py
│   ├── test_file_manager.py
│   ├── test_review_client.py
│   ├── test_review_context.py
│   ├── test_review_details.py
│   ├── test_review_summary.py
│   ├── test_sessions.py
│   ├── test_sources.py
│   └── test_vector_store.py
│
├── scripts/
│   ├── reembed.py               # Re-embeds stored chunks after an embedding change
│   └── smoke_test.py            # Manual smoke test (needs real servers running)
│
├── uploads/                    # Uploaded files
│   └── media/                  # YouTube audio
│
├── data/                       # Saved chat sessions (sessions.json)
│
└── chroma_db/                  # Vector database
```

---

## 💡 Usage Examples

### Example 1: Upload and Chat
```python
import requests

BASE_URL = 'http://localhost:5001'

# Upload PDF
with open('document.pdf', 'rb') as f:
    requests.post(f'{BASE_URL}/upload/file', files={'file': f})

# Create session
response = requests.post(f'{BASE_URL}/chat/new')
session_id = response.json()['session_id']

# Ask question
response = requests.post(f'{BASE_URL}/chat', json={
    'question': 'What is this document about?',
    'session_id': session_id
})
print(response.json()['answer'])
```

### Example 2: YouTube Chat
```python
# Upload YouTube video
requests.post(f'{BASE_URL}/upload/youtube', 
    json={'url': 'https://www.youtube.com/watch?v=...'})

# Chat with transcription
response = requests.post(f'{BASE_URL}/chat',
    json={'question': 'Summarize the video'})
```

### Example 3: cURL Commands
```bash
# Upload web page
curl -X POST http://localhost:5001/upload/web \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/Python_(programming_language)"}'

# Chat
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python?"}'

# Get stats
curl http://localhost:5001/stats
```

---

## 🧪 Testing

### Unit/Integration Tests (mocked — no servers, no Ollama, no Whisper model needed)
```bash
pytest
```

The same tests run on every pull request and push to `master` through GitHub Actions
(`.github/workflows/ci.yml`, Python 3.12). CI does not install Whisper or PyTorch, because
the speech model is loaded the first time a YouTube upload needs it and no test transcribes audio.

### Manual Smoke Test (needs Ollama + this server running)
```bash
python scripts/smoke_test.py
```

### Manual Testing

Use Postman, cURL, or Python requests to test endpoints. See [Usage Examples](#-usage-examples).

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Ollama call failed 404" | Run `ollama pull llama3.2`, `ollama pull qwen2.5-coder:7b` and `ollama pull nomic-embed-text` |
| "Connection refused" | Ensure both Ollama (`ollama serve`) and Flask (`python main.py`) are running |
| "415 Unsupported Media Type" | Add header: `-H "Content-Type: application/json"` |
| "FFmpeg not found" | Install FFmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux) |
| Slow responses | Use smaller model (`phi3`), reduce chunk size, or decrease `k` parameter |
| Too many old chats | POST to `/chat/cleanup` to delete sessions not used for a day (pinned ones stay) |

---

## ⚡ Performance Tips

1. **Choose Right Model**
   - Fast: `phi3`
   - Balanced: `llama3.2:1b`
   - Best: `llama3.2` (default)

2. **Optimize Chunk Size**
```python
   # In document_processor.py
   chunk_size=500,  # Smaller = faster
   chunk_overlap=50
```

3. **Reduce Retrieved Chunks**
```python
   {"question": "...", "k": 3}  # Default is 5
```

4. **Regular Cleanup**
```bash
   # Delete sessions not used for a day (pinned ones stay)
   curl -X POST http://localhost:5001/chat/cleanup \
     -H "Content-Type: application/json" \
     -d '{"inactive_hours": 24}'
```

---

## 🚧 Future Enhancements

**Planned Features:**
- [ ] User authentication & authorization
- [ ] Database persistence (PostgreSQL)
- [ ] Frontend UI (React/Vue)
- [ ] More file formats (DOCX, PPTX, CSV)
- [ ] Streaming responses
- [ ] Document update/deletion
- [ ] Export conversations
- [ ] Docker containerization
- [ ] Analytics dashboard

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

**Code Style:**
- Follow PEP 8
- Add docstrings
- Include type hints
- Write tests

---

## 👨‍💻 Author

**Namrata Roy**
- GitHub: [@Rnamrata](https://github.com/Rnamrata/llm-chatbot.git)
- Email: roy.namrata.cse@gmail.com

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - RAG framework
- [Ollama](https://ollama.ai/) - Local LLM
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Whisper](https://github.com/openai/whisper) - Transcription

---

**Built using Python, LangChain, and Ollama**
