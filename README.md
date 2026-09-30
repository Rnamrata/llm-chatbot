# 🤖 RAG Chatbot System

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot system that allows you to chat with your documents using local LLMs. Upload files, YouTube videos, or web pages, and have intelligent conversations with the content.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
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
- 💾 **Session Management** - Multiple concurrent conversations

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
2. **Chat:** Question → Retrieve Context → LLM → Answer with Sources
3. **Code Review:** Code file → validated & stored → sent to the configured review service → findings stored → chat session scoped to that review via `review_id`

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask 3.0+ |
| LLM | Ollama (llama3.2) |
| Embeddings | nomic-embed-text |
| Vector DB | ChromaDB |
| Framework | LangChain |
| PDF Processing | PyPDF2 |
| Audio | OpenAI Whisper |
| Video | yt-dlp |

---

## 📦 Installation

### Prerequisites

- Python 3.8+
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
| `REVIEW_LLM_TEMPERATURE` | `0.2` | Lower temperature used for code-review chat mode |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHROMA_PERSIST_DIR` / `CHROMA_COLLECTION` | `./chroma_db` / `my_documents` | Vector store location/collection |
| `RETRIEVAL_K` | `5` | Default number of chunks retrieved per query |
| `UPLOAD_DIR` / `MEDIA_DIR` | `./uploads` / `./uploads/media` | Where uploaded files/audio are saved |
| `MAX_UPLOAD_MB` | `20` | Max upload size |
| `WHISPER_MODEL` | `base` | Whisper model for YouTube transcription |
| `REVIEW_SERVICE_URL` / `REVIEW_TIMEOUT_SECONDS` | `http://localhost:4000` / `60` | Where the code review service is running |

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

---

## 📡 API Reference

**Base URL:** `http://localhost:5001`

### Upload Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/upload/file` | POST | `file: <file>` | Upload PDF/TXT/MD |
| `/upload/youtube` | POST | `{"url": "..."}` | Transcribe YouTube video |
| `/upload/web` | POST | `{"url": "..."}` | Scrape web page |
| `/upload/code` | POST | `file: <code file>`, `session_id` | Upload code for review without attaching it to a chat session |

### Code Review Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/review` | POST | `file: <code file>`, `session_id` (optional) | Review a code file, store the code + findings, and attach the review to a chat session (creates one if `session_id` is omitted) |
| `/review/{review_id}` | GET | - | Reload a previously stored review's code and findings |

### Chat Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/chat/new` | POST | - | Create new session |
| `/chat` | POST | `{"question": "...", "session_id": "..."}` | Send message |
| `/chat/history/{id}` | GET | - | Get conversation history |
| `/chat/session/{id}` | GET | - | Get session info |
| `/chat/clear/{id}` | DELETE | - | Clear session |
| `/chat/sessions` | GET | - | List all sessions |
| `/chat/cleanup` | POST | `{"inactive_hours": 24}` | Remove old sessions |

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
│   └── review_client.py        # Code review service client
│
├── src/config.py                # Central configuration (reads .env)
├── .env.example                 # Environment variable template
├── pytest.ini                   # pytest configuration
│
├── tests/                       # Automated tests (pytest, fully mocked)
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_document_processor.py
│   └── test_review_client.py
│
├── scripts/
│   └── smoke_test.py            # Manual smoke test (needs real servers running)
│
├── uploads/                    # Uploaded files
│   └── media/                  # YouTube audio
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

### Unit/Integration Tests (mocked — no servers needed)
```bash
pytest
```

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
| "Ollama call failed 404" | Run `ollama pull llama3.2` and `ollama pull nomic-embed-text` |
| "Connection refused" | Ensure both Ollama (`ollama serve`) and Flask (`python main.py`) are running |
| "415 Unsupported Media Type" | Add header: `-H "Content-Type: application/json"` |
| "FFmpeg not found" | Install FFmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux) |
| Slow responses | Use smaller model (`phi3`), reduce chunk size, or decrease `k` parameter |
| Out of memory | Clean up sessions: POST to `/chat/cleanup` |

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
   # Clean up old sessions daily
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