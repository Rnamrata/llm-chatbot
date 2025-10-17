# 🤖 Dual-Mode RAG Chatbot System

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot system with dual functionality:
1. **Document Q&A Mode** - Chat with your documents using local LLMs
2. **Code Review Mode** - AI-powered code analysis and review

Upload documents, code files, YouTube videos, or web pages, and have intelligent conversations or get comprehensive code reviews.

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

### Document Q&A Mode
- 📄 **Multi-Source Document Upload** - PDF, TXT, Markdown, YouTube, Web pages
- 🧠 **Intelligent Conversation** - Context-aware with conversation history
- 🔍 **Semantic Search** - Vector-based retrieval with source attribution
- 💾 **Session Management** - Multiple concurrent conversations
- 📊 **Source Attribution** - Track which documents answer came from

### Code Review Mode
- 💻 **Multi-Language Support** - Python, JavaScript, TypeScript, Java, Go, C/C++, Rust, Ruby, PHP, Swift, Kotlin, C#, and 10+ more
- 🔍 **Quick Review** - Fast critical issue detection
- 📋 **Comprehensive Review** - Detailed analysis across 6 categories (quality, security, performance, best practices, bugs, testing)
- 🔒 **Security Analysis** - OWASP Top 10 vulnerability detection
- ⚡ **Performance Review** - Algorithm complexity and optimization suggestions
- 🐛 **Bug Detection** - Potential bug identification with fixes
- 💡 **Code Explanation** - Plain language code explanations
- 🎯 **Improvement Suggestions** - Refactoring and enhancement recommendations
- 📏 **Complexity Metrics** - LOC, functions, classes, cyclomatic complexity
- 🏗️ **Syntax-Aware Chunking** - Preserves function/class boundaries

### Technical Features
- 🏗️ **Modular Architecture** - Clean separation of concerns with dependency injection
- 🔌 **RESTful API** - 20+ endpoints for documents and code
- 🎯 **Local LLM** - Privacy-focused with Ollama (no data sent to cloud)
- 🔄 **Hybrid System** - Both modes share same vector database for cross-referencing
- 🚀 **Automatic Pipeline** - One-call upload to processing to storage
- 📊 **Monitoring** - Built-in stats and health checks
- 🛡️ **Error Handling** - Comprehensive error management

---

## 🏛️ Architecture
```
Client → Flask API → FileManager/ChatSession → DocumentProcessor/LLMManager 
                                            → VectorStore → Ollama
```

**Data Flow:**
1. **Upload:** Document → Extract → Chunk → Embed → Store
2. **Chat:** Question → Retrieve Context → LLM → Answer with Sources

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

### requirements.txt
```txt
flask==3.0.0
flask-cors==4.0.0
langchain==0.1.0
langchain-community==0.0.10
chromadb==0.4.22
openai-whisper==20231117
yt-dlp==2023.12.30
PyPDF2==3.0.1
requests==2.31.0
numpy==1.24.3
```

---

## 🚀 Quick Start
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Flask Server
cd llm-chatbot
source .venv/bin/activate
python main.py

# Terminal 3: Test
python test_system.py
```

Server will be available at: `http://localhost:5001`

---

## 📡 API Reference

**Base URL:** `http://localhost:5001`

**Full API Documentation:** See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete details with examples.

### Document Upload Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/upload/file` | POST | `file: <file>` | Upload document (PDF/TXT/MD) or code file |
| `/upload/youtube` | POST | `{"url": "..."}` | Transcribe YouTube video |
| `/upload/web` | POST | `{"url": "..."}` | Scrape web page |
| `/upload/code` | POST | `file: <code_file>` | Upload code file with complexity analysis |

### Code Review Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/review/quick` | POST | `file: <code>` | Quick review (critical issues only) |
| `/review/comprehensive` | POST | `file: <code>` or JSON | Detailed code review with context |
| `/review/security` | POST | `file: <code>` | Security vulnerability analysis |
| `/review/performance` | POST | `file: <code>` | Performance optimization review |
| `/review/explain` | POST | `file: <code>` or JSON | Explain code in plain language |
| `/review/bugs` | POST | `file: <code>` or JSON | Detect potential bugs |
| `/review/improve` | POST | `file: <code>` or JSON | Suggest improvements |

### Chat Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/chat/new` | POST | - | Create new session |
| `/chat` | POST | `{"question": "...", "session_id": "..."}` | Ask about documents or code |
| `/chat/history/{id}` | GET | - | Get conversation history |
| `/chat/session/{id}` | GET | - | Get session info |
| `/chat/clear/{id}` | DELETE | - | Clear session |
| `/chat/sessions` | GET | - | List all sessions |
| `/chat/cleanup` | POST | `{"inactive_hours": 24}` | Remove old sessions |

### Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Database statistics |

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
├── main.py                              # Flask app (500+ lines, 20+ endpoints)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── API_DOCUMENTATION.md                 # Complete API reference
├── TECHNICAL_DOCUMENTATION.md           # In-depth code explanations
│
├── src/modules/
│   ├── file_manager.py                  # Upload orchestration (260 lines)
│   ├── document_processor.py            # Document text extraction & chunking (75 lines)
│   ├── code_parser.py                   # Code parsing & syntax-aware chunking (405 lines)
│   ├── code_review_prompts.py           # Review prompt templates (402 lines)
│   ├── vector_store_and_embedding.py    # Vector DB operations (54 lines)
│   ├── llm_manager.py                   # LLM chains & code review (380+ lines)
│   └── chat_session.py                  # Session management (260 lines)
│
├── test/
│   └── test_system.py                   # System integration tests
│
├── uploads/                             # User uploaded files
│   └── media/                           # YouTube audio files
│
└── chroma_db/                           # ChromaDB vector database
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

### Example 3: Code Review Workflow
```python
# Upload code file
with open('app.py', 'rb') as f:
    response = requests.post(f'{BASE_URL}/upload/code', files={'file': f})
    print(response.json()['complexity'])  # Get metrics

# Quick review
with open('app.py', 'rb') as f:
    response = requests.post(f'{BASE_URL}/review/quick', files={'file': f})
    print(response.json()['review'])

# Security review
with open('app.py', 'rb') as f:
    response = requests.post(f'{BASE_URL}/review/security', files={'file': f})
    print(response.json()['review'])

# Ask questions about code via chat
response = requests.post(f'{BASE_URL}/chat', json={
    'question': 'How can I optimize the database queries in app.py?'
})
print(response.json()['answer'])
```

### Example 4: cURL Commands
```bash
# Upload document
curl -X POST http://localhost:5001/upload/file \
  -F "file=@document.pdf"

# Upload code for review
curl -X POST http://localhost:5001/upload/code \
  -F "file=@app.py"

# Quick code review
curl -X POST http://localhost:5001/review/quick \
  -F "file=@app.py"

# Comprehensive review with JSON
curl -X POST http://localhost:5001/review/comprehensive \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello():\n    print(\"Hello\")",
    "filename": "test.py",
    "question": "Is this function well-written?"
  }'

# Chat with documents or code
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python?"}'

# Get stats
curl http://localhost:5001/stats
```

---

## 🧪 Testing

### Automated Tests
```bash
python test_system.py
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

## 📚 Documentation

This project includes comprehensive documentation:

| File | Description |
|------|-------------|
| **README.md** (this file) | Quick start guide and overview |
| **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** | Complete API reference with examples for all 20+ endpoints |
| **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** | In-depth technical guide explaining every code component, design patterns, data flows, and architecture |

**What's in TECHNICAL_DOCUMENTATION.md:**
- Detailed module-by-module code explanations (1,800+ lines)
- Complete data flow diagrams
- Design patterns used (Dependency Injection, Strategy, Template Method, Facade, Repository)
- SOLID principles implementation
- Configuration & setup guide
- Extension guide (add new languages, review types, vector DBs, file types)
- Performance optimization strategies
- Security considerations
- Testing approaches

---

## 🚧 Future Enhancements

**Recently Added:**
- [x] Code review functionality with multi-language support
- [x] Syntax-aware code chunking
- [x] Comprehensive code analysis (security, performance, bugs)
- [x] Code complexity metrics

**Planned Features:**
- [ ] User authentication & authorization
- [ ] Database persistence (PostgreSQL for conversations)
- [ ] Frontend UI (React/Vue)
- [ ] More file formats (DOCX, PPTX, CSV, Excel)
- [ ] Streaming responses (real-time LLM output)
- [ ] Document update/deletion
- [ ] Export conversations (JSON, Markdown)
- [ ] Docker containerization
- [ ] Analytics dashboard
- [ ] Git integration (review pull requests)
- [ ] IDE extensions (VS Code, IntelliJ)

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