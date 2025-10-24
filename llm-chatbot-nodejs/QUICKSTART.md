# Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies

```bash
cd llm-chatbot-nodejs
npm install
```

### 2. Start Required Services

**Ollama** (Terminal 1):
```bash
# Install models if not already done
ollama pull llama3.2
ollama pull nomic-embed-text

# Ollama runs as a service, verify it's running
ollama list
```

**ChromaDB** (Terminal 2):
```bash
# Option 1: Docker (recommended)
docker run -d -p 8000:8000 chromadb/chroma

# Option 2: Python
pip install chromadb
chroma run --path ./chroma_db
```

### 3. Configure Environment

```bash
cp .env.example .env
# Default values should work fine for local development
```

### 4. Start the Server

```bash
# Development mode with auto-reload
npm run dev

# Or production mode
npm start
```

Server will be available at: **http://localhost:5001**

---

## Test the System

### 1. Check Health

```bash
curl http://localhost:5001/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "RAG System",
  "version": "1.0",
  "llm_model": "llama3.2"
}
```

### 2. Upload a Test Document

Create a test file `test.txt`:
```bash
echo "Node.js is a JavaScript runtime built on Chrome's V8 engine." > test.txt
```

Upload it:
```bash
curl -X POST http://localhost:5001/upload/file \
  -F "file=@test.txt"
```

### 3. Ask a Question

```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Node.js?"
  }'
```

### 4. Review Code

Create a test code file `example.js`:
```javascript
function add(a, b) {
  return a + b;
}
```

Review it:
```bash
curl -X POST http://localhost:5001/review/quick \
  -F "file=@example.js"
```

---

## Project Structure Overview

```
llm-chatbot-nodejs/
├── src/
│   ├── server.js              # Main Express app
│   └── modules/               # Core modules
│       ├── codeParser.js      # Code analysis
│       ├── llmManager.js      # LLM interactions
│       ├── chatSession.js     # Session management
│       └── ...
├── uploads/                   # File storage
├── package.json              # Dependencies
└── .env                      # Configuration
```

---

## Common Tasks

### Upload Different File Types

**PDF Document:**
```bash
curl -X POST http://localhost:5001/upload/file -F "file=@document.pdf"
```

**Code File:**
```bash
curl -X POST http://localhost:5001/upload/code -F "file=@app.js"
```

**YouTube Video:**
```bash
curl -X POST http://localhost:5001/upload/youtube \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=VIDEO_ID"}'
```

**Web Page:**
```bash
curl -X POST http://localhost:5001/upload/web \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

### Code Review Types

**Comprehensive Review:**
```bash
curl -X POST http://localhost:5001/review/comprehensive -F "file=@code.js"
```

**Security Scan:**
```bash
curl -X POST http://localhost:5001/review/security -F "file=@code.js"
```

**Performance Analysis:**
```bash
curl -X POST http://localhost:5001/review/performance -F "file=@code.js"
```

**Bug Detection:**
```bash
curl -X POST http://localhost:5001/review/bugs -F "file=@code.js"
```

### Session Management

**Create New Session:**
```bash
curl -X POST http://localhost:5001/chat/new
# Returns: {"session_id": "abc-123", ...}
```

**Chat with Session:**
```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain this in detail",
    "session_id": "abc-123"
  }'
```

**View History:**
```bash
curl http://localhost:5001/chat/history/abc-123
```

**List All Sessions:**
```bash
curl http://localhost:5001/chat/sessions
```

---

## Troubleshooting

### Port 5001 Already in Use

Change port in `.env`:
```env
PORT=5002
```

### ChromaDB Not Connecting

Check if ChromaDB is running:
```bash
curl http://localhost:8000/api/v1/heartbeat
```

### Ollama Not Responding

Verify Ollama service:
```bash
ollama list
curl http://localhost:11434/api/tags
```

### Module Not Found Errors

Reinstall dependencies:
```bash
rm -rf node_modules package-lock.json
npm install
```

---

## Next Steps

1. ✅ **Read the full README.md** for detailed API documentation
2. ✅ **Explore the source code** in `src/modules/`
3. ✅ **Try different review types** on your own code
4. ✅ **Build a frontend** using the REST API
5. ✅ **Add custom prompts** in `codeReviewPrompts.js`

---

## Development Tips

**Watch Mode:**
```bash
npm run dev  # Auto-reloads on file changes
```

**Testing:**
```bash
npm test
```

**Linting:**
```bash
# Add to package.json scripts if needed
npm run lint
```

**Debugging:**
Add to your code:
```javascript
console.log('Debug:', variable);
```

Or use Node.js inspector:
```bash
node --inspect src/server.js
```

---

## Performance Tips

1. **Chunk Size**: Adjust in `documentProcessor.js` (default: 1000 chars)
2. **Retrieval Count**: Change `k` parameter in chat requests (default: 5)
3. **LLM Temperature**: Modify in `.env` (default: 0.7)
4. **Session Cleanup**: Run periodically to free memory

---

## Comparison with Python Version

| Feature | Python | Node.js |
|---------|--------|---------|
| Web Framework | Flask | Express.js |
| File Uploads | request.files | Multer |
| LangChain | Python | JavaScript |
| YouTube | yt-dlp | ytdl-core |
| Whisper | openai-whisper | @xenova/transformers |
| Web Scraping | BeautifulSoup | Cheerio |

**All API endpoints remain the same!** 🎉

---

Happy Coding! 🚀
