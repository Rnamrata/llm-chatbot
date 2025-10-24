# LLM Chatbot - Node.js Version

A powerful RAG (Retrieval-Augmented Generation) chatbot system with comprehensive code review capabilities, built with Node.js and Express.

**Converted from Python to Node.js**

## 📚 Comprehensive Documentation

**New to the system?** Start with [QUICKSTART.md](QUICKSTART.md) for a 5-minute setup guide.

**Complete Documentation**:
- 📖 **[Documentation Index](docs/README.md)** - Start here to navigate all docs
- 🔍 **[How It Works](docs/HOW_IT_WORKS.md)** - Complete system explanation with examples
- 🏗️ **[Architecture](docs/ARCHITECTURE.md)** - System design and component interactions
- 💻 **[Code Explained](docs/CODE_EXPLAINED.md)** - Line-by-line code walkthrough
- 🌐 **[API Documentation](docs/API_DOCUMENTATION.md)** - Complete REST API reference
- ⚡ **[Quick Start](QUICKSTART.md)** - Get running in 5 minutes

**Over 5,000 lines of detailed documentation covering every aspect of the system!**

## Features

- **Document Processing**: Upload and query PDF, TXT, and Markdown files
- **Code Review**: Comprehensive code analysis with multiple review types
- **YouTube Integration**: Transcribe and query YouTube videos
- **Web Scraping**: Extract and query web page content
- **Conversational Memory**: Persistent chat sessions with history
- **Vector Search**: ChromaDB for semantic document retrieval
- **Local LLM**: Privacy-focused using Ollama (no cloud dependencies)

## Architecture

```
llm-chatbot-nodejs/
├── src/
│   ├── server.js                          # Main Express server
│   ├── modules/
│   │   ├── codeParser.js                  # Code parsing & analysis
│   │   ├── codeReviewPrompts.js           # Review prompt templates
│   │   ├── chatSession.js                 # Session management
│   │   ├── documentProcessor.js           # Document processing
│   │   ├── fileManager.js                 # File upload handling
│   │   ├── llmManager.js                  # LLM operations
│   │   └── vectorStoreAndEmbedding.js     # Vector database
├── uploads/                               # Uploaded files
│   └── media/                            # YouTube audio files
├── tests/                                # Test files
├── package.json                          # Dependencies
├── .env.example                          # Environment template
└── README.md                             # This file
```

## Prerequisites

- Node.js >= 18.0.0
- [Ollama](https://ollama.ai/) with `llama3.2` and `nomic-embed-text` models
- [ChromaDB](https://www.trychroma.com/) server running on port 8000

### Install Ollama Models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Start ChromaDB

```bash
# Using Docker
docker run -d -p 8000:8000 chromadb/chroma

# Or install locally
pip install chromadb
chroma run --path ./chroma_db
```

## Installation

1. **Clone or navigate to the project**

```bash
cd llm-chatbot-nodejs
```

2. **Install dependencies**

```bash
npm install
```

3. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Create upload directories**

```bash
mkdir -p uploads/media
```

## Running the Server

### Development Mode

```bash
npm run dev
```

### Production Mode

```bash
npm start
```

The server will start on `http://localhost:5001`

## API Endpoints

### Upload Endpoints

- `POST /upload/file` - Upload document (PDF, TXT, MD)
- `POST /upload/youtube` - Process YouTube video
- `POST /upload/web` - Scrape web page
- `POST /upload/code` - Upload code for review

### Code Review Endpoints

- `POST /review/quick` - Quick critical issues review
- `POST /review/comprehensive` - Full detailed analysis
- `POST /review/security` - Security vulnerability scan
- `POST /review/performance` - Performance optimization
- `POST /review/explain` - Code explanation
- `POST /review/bugs` - Bug detection
- `POST /review/improve` - Improvement suggestions

### Chat Endpoints

- `POST /chat` - Ask questions
- `POST /chat/new` - Create new session
- `GET /chat/session/:id` - Get session info
- `GET /chat/history/:id` - Get conversation history
- `DELETE /chat/clear/:id` - Clear session
- `GET /chat/sessions` - List all sessions
- `POST /chat/cleanup` - Remove inactive sessions

### Utility Endpoints

- `GET /health` - Health check
- `GET /stats` - Database statistics

## Usage Examples

### Upload a Document

```bash
curl -X POST http://localhost:5001/upload/file \
  -F "file=@document.pdf"
```

### Ask a Question

```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "session_id": "your-session-id"
  }'
```

### Review Code

```bash
curl -X POST http://localhost:5001/review/comprehensive \
  -F "file=@app.js" \
  -F "question=Review this code for best practices"
```

### Process YouTube Video

```bash
curl -X POST http://localhost:5001/upload/youtube \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=..."}'
```

## Supported Programming Languages

Python, JavaScript, TypeScript, Java, Go, C/C++, Rust, Ruby, PHP, Swift, Kotlin, C#, HTML, CSS, SQL, Shell/Bash

## Testing

```bash
npm test
```

## Key Differences from Python Version

- **Express.js** instead of Flask
- **Multer** for file uploads instead of Flask's request.files
- **LangChain.js** instead of LangChain Python
- **ytdl-core** instead of yt-dlp
- **@xenova/transformers** for Whisper instead of openai-whisper
- **Cheerio** instead of BeautifulSoup
- **ES Modules** (import/export) instead of CommonJS

## Development

### Project Structure

- `src/modules/` - Core business logic modules
- `src/server.js` - Express application setup
- `uploads/` - File storage directory
- `tests/` - Test suites

### Adding New Features

1. Create module in `src/modules/`
2. Add routes in `src/server.js`
3. Update this README

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| PORT | Server port | 5001 |
| OLLAMA_BASE_URL | Ollama API URL | http://localhost:11434 |
| OLLAMA_MODEL | LLM model name | llama3.2 |
| CHROMA_URL | ChromaDB URL | http://localhost:8000 |
| LLM_TEMPERATURE | Response randomness | 0.7 |

## Troubleshooting

### ChromaDB Connection Error

Ensure ChromaDB is running:
```bash
docker ps | grep chroma
```

### Ollama Model Not Found

Pull required models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Port Already in Use

Change PORT in `.env` file or stop the conflicting process.

## Performance Considerations

- **Chunking**: Documents are split into 1000-character chunks
- **Vector Search**: Uses MMR (Maximum Marginal Relevance) for diverse results
- **Memory**: Conversation history is stored in-memory
- **File Size**: Default max upload size is 10MB

## Security Notes

- Input validation on all file uploads
- File type restrictions enforced
- No credentials stored in code
- Local LLM processing (no data sent to cloud)

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Acknowledgments

- Built with LangChain.js
- Powered by Ollama
- Vector storage by ChromaDB
- Transcription by Xenova Transformers

## Support

For issues and questions:
- Check existing issues
- Create a new issue with details
- Include error logs and system info

---

**Original Python Version**: See `/llm-chatbot/` directory
**Node.js Version**: Current directory
