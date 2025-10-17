# API Documentation - Dual-Mode Chatbot (Documents & Code Review)

Base URL: `http://localhost:5001`

## Table of Contents
- [Document Upload Endpoints](#document-upload-endpoints)
- [Code Upload Endpoints](#code-upload-endpoints)
- [Code Review Endpoints](#code-review-endpoints)
- [Chat Endpoints](#chat-endpoints)
- [Utility Endpoints](#utility-endpoints)

---

## Document Upload Endpoints

### 1. Upload Document File
**Endpoint:** `POST /upload/file`

**Description:** Upload PDF, TXT, or MD files for document Q&A

**Request:**
- **Type:** multipart/form-data
- **Parameters:**
  - `file`: Document file (PDF, TXT, MD)

**Response:**
```json
{
  "success": true,
  "message": "File uploaded and processed successfully",
  "filename": "document.pdf",
  "file_type": "document",
  "chunks_created": 15
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/upload/file \
  -F "file=@document.pdf"
```

---

### 2. Upload YouTube Video
**Endpoint:** `POST /upload/youtube`

**Description:** Download, transcribe, and index YouTube video content

**Request:**
- **Type:** application/json or form-data
- **Parameters:**
  - `url`: YouTube video URL

**Response:**
```json
{
  "success": true,
  "message": "YouTube video processed and stored successfully",
  "url": "https://youtube.com/watch?v=...",
  "transcription_file": "video_transcription.txt",
  "chunks_created": 25
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/upload/youtube \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=example"}'
```

---

### 3. Upload Web Page
**Endpoint:** `POST /upload/web`

**Description:** Scrape and index web page content

**Request:**
- **Type:** application/json or form-data
- **Parameters:**
  - `url`: Web page URL

**Response:**
```json
{
  "success": true,
  "message": "Web page processed and stored successfully",
  "url": "https://example.com/article",
  "filename": "article_name.txt",
  "chunks_created": 12
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/upload/web \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

---

## Code Upload Endpoints

### 4. Upload Code File
**Endpoint:** `POST /upload/code`

**Description:** Upload code file for review with complexity analysis

**Supported Languages:**
- Python (.py)
- JavaScript (.js)
- TypeScript (.ts, .tsx)
- Java (.java)
- Go (.go)
- C/C++ (.c, .cpp, .h, .hpp)
- Rust (.rs)
- Ruby (.rb)
- PHP (.php)
- Swift (.swift)
- Kotlin (.kt)
- C# (.cs)
- And more...

**Request:**
- **Type:** multipart/form-data
- **Parameters:**
  - `file`: Code file

**Response:**
```json
{
  "success": true,
  "message": "Code file uploaded for review successfully",
  "filename": "app.py",
  "language": "python",
  "chunks_created": 8,
  "complexity": {
    "lines_of_code": 150,
    "num_functions": 5,
    "num_classes": 2,
    "num_imports": 8,
    "cyclomatic_complexity": 23
  }
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/upload/code \
  -F "file=@app.py"
```

---

## Code Review Endpoints

### 5. Quick Code Review
**Endpoint:** `POST /review/quick`

**Description:** Fast code review focusing on critical issues

**Request:**
- **Type:** multipart/form-data
- **Parameters:**
  - `file`: Code file to review

**Response:**
```json
{
  "success": true,
  "filename": "app.py",
  "language": "python",
  "review_type": "quick",
  "review": "Critical Issues Found:\n1. Line 45: SQL injection vulnerability..."
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/review/quick \
  -F "file=@app.py"
```

---

### 6. Comprehensive Code Review
**Endpoint:** `POST /review/comprehensive`

**Description:** Detailed code review with context from codebase

**Request (File Upload):**
- **Type:** multipart/form-data
- **Parameters:**
  - `file`: Code file to review
  - `question`: (Optional) Specific review focus

**Request (JSON):**
- **Type:** application/json
- **Parameters:**
  - `code`: Code content as string
  - `filename`: File name
  - `question`: (Optional) Specific review focus

**Response:**
```json
{
  "success": true,
  "filename": "app.py",
  "language": "python",
  "review_type": "comprehensive",
  "review": "Comprehensive Review:\n\n1. Code Quality...",
  "context_used": 3
}
```

**Example (curl - File):**
```bash
curl -X POST http://localhost:5001/review/comprehensive \
  -F "file=@app.py" \
  -F "question=Focus on database operations"
```

**Example (curl - JSON):**
```bash
curl -X POST http://localhost:5001/review/comprehensive \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello():\n    print(\"Hello\")",
    "filename": "test.py",
    "question": "Is this function well-written?"
  }'
```

---

### 7. Security Review
**Endpoint:** `POST /review/security`

**Description:** Security-focused code analysis

**Request:**
- **Type:** multipart/form-data
- **Parameters:**
  - `file`: Code file to review

**Response:**
```json
{
  "success": true,
  "filename": "app.py",
  "language": "python",
  "review_type": "security",
  "review": "Security Analysis:\n\nCRITICAL:\n- SQL Injection on line 45..."
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/review/security \
  -F "file=@app.py"
```

---

### 8. Performance Review
**Endpoint:** `POST /review/performance`

**Description:** Performance-focused code analysis

**Request:**
- **Type:** multipart/form-data
- **Parameters:**
  - `file`: Code file to review

**Response:**
```json
{
  "success": true,
  "filename": "app.py",
  "language": "python",
  "review_type": "performance",
  "review": "Performance Analysis:\n\n1. O(n²) complexity on line 30..."
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/review/performance \
  -F "file=@app.py"
```

---

### 9. Explain Code
**Endpoint:** `POST /review/explain`

**Description:** Get clear explanation of what code does

**Request (File Upload):**
- **Type:** multipart/form-data
- **Parameters:**
  - `file`: Code file
  - `question`: (Optional) Specific question

**Request (JSON):**
- **Type:** application/json
- **Parameters:**
  - `code`: Code content
  - `filename`: File name
  - `question`: (Optional) Specific question

**Response:**
```json
{
  "success": true,
  "filename": "app.py",
  "language": "python",
  "explanation": "This code implements a user authentication system..."
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/review/explain \
  -F "file=@app.py" \
  -F "question=What does the login function do?"
```

---

### 10. Detect Bugs
**Endpoint:** `POST /review/bugs`

**Description:** Analyze code for potential bugs

**Request (File Upload):**
- **Type:** multipart/form-data
- **Parameters:**
  - `file`: Code file
  - `issue`: (Optional) Reported issue description

**Request (JSON):**
- **Type:** application/json
- **Parameters:**
  - `code`: Code content
  - `filename`: File name
  - `issue`: (Optional) Reported issue

**Response:**
```json
{
  "success": true,
  "filename": "app.py",
  "language": "python",
  "bug_analysis": "Potential Bugs Found:\n\n1. Off-by-one error on line 42..."
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/review/bugs \
  -F "file=@app.py" \
  -F "issue=Getting IndexError when processing lists"
```

---

### 11. Suggest Improvements
**Endpoint:** `POST /review/improve`

**Description:** Get specific code improvement suggestions

**Request (File Upload):**
- **Type:** multipart/form-data
- **Parameters:**
  - `file`: Code file
  - `goal`: (Optional) Improvement goal

**Request (JSON):**
- **Type:** application/json
- **Parameters:**
  - `code`: Code content
  - `filename`: File name
  - `goal`: (Optional) Improvement goal

**Response:**
```json
{
  "success": true,
  "filename": "app.py",
  "language": "python",
  "suggestions": "Improvement Suggestions:\n\n1. Use list comprehension instead of loop..."
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/review/improve \
  -F "file=@app.py" \
  -F "goal=Make it more Pythonic"
```

---

## Chat Endpoints

### 12. Chat with Documents
**Endpoint:** `POST /chat`

**Description:** Ask questions about uploaded documents or code

**Request:**
- **Type:** application/json
- **Parameters:**
  - `question`: User's question (required)
  - `session_id`: Session ID for continuity (optional)
  - `k`: Number of chunks to retrieve (optional, default: 5)

**Response:**
```json
{
  "success": true,
  "answer": "Based on the documents, the answer is...",
  "sources": [
    {
      "content": "Relevant content preview...",
      "metadata": {
        "source": "document.pdf",
        "type": "file_upload"
      }
    }
  ],
  "num_sources": 5,
  "session_id": "abc-123",
  "message_count": 3
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "session_id": "my-session-123",
    "k": 5
  }'
```

---

### 13. Create New Chat Session
**Endpoint:** `POST /chat/new`

**Description:** Start a new chat session

**Response:**
```json
{
  "session_id": "abc-def-123",
  "message": "New chat session created"
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/chat/new
```

---

### 14. Get Session Info
**Endpoint:** `GET /chat/session/<session_id>`

**Description:** Get metadata about a session

**Response:**
```json
{
  "exists": true,
  "created_at": "2024-01-15T10:30:00",
  "last_activity": "2024-01-15T11:45:00",
  "message_count": 5
}
```

**Example (curl):**
```bash
curl http://localhost:5001/chat/session/abc-def-123
```

---

### 15. Get Chat History
**Endpoint:** `GET /chat/history/<session_id>`

**Description:** Retrieve conversation history

**Response:**
```json
{
  "history": [
    {"question": "What is...", "answer": "The answer is..."},
    {"question": "How does...", "answer": "It works by..."}
  ],
  "length": 2,
  "message_count": 2
}
```

**Example (curl):**
```bash
curl http://localhost:5001/chat/history/abc-def-123
```

---

### 16. Clear Chat History
**Endpoint:** `DELETE /chat/clear/<session_id>`

**Description:** Clear a session's history

**Response:**
```json
{
  "success": true,
  "message": "Chat history cleared for session abc-def-123"
}
```

**Example (curl):**
```bash
curl -X DELETE http://localhost:5001/chat/clear/abc-def-123
```

---

### 17. List All Sessions
**Endpoint:** `GET /chat/sessions`

**Description:** Get all active sessions

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "abc-123",
      "created_at": "2024-01-15T10:00:00",
      "last_activity": "2024-01-15T11:00:00",
      "message_count": 3
    }
  ],
  "total_sessions": 1
}
```

**Example (curl):**
```bash
curl http://localhost:5001/chat/sessions
```

---

### 18. Cleanup Inactive Sessions
**Endpoint:** `POST /chat/cleanup`

**Description:** Remove inactive sessions

**Request:**
- **Type:** application/json
- **Parameters:**
  - `inactive_hours`: Hours of inactivity threshold (default: 24)

**Response:**
```json
{
  "success": true,
  "cleaned": 3,
  "remaining": 2,
  "message": "Cleaned up 3 inactive sessions"
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:5001/chat/cleanup \
  -H "Content-Type: application/json" \
  -d '{"inactive_hours": 12}'
```

---

## Utility Endpoints

### 19. Get Statistics
**Endpoint:** `GET /stats`

**Description:** Get system statistics

**Response:**
```json
{
  "total_chunks": 150,
  "total_sessions": 5,
  "status": "ready",
  "message": "Vector database contains 150 chunks"
}
```

**Example (curl):**
```bash
curl http://localhost:5001/stats
```

---

### 20. Health Check
**Endpoint:** `GET /health`

**Description:** Check system health

**Response:**
```json
{
  "status": "healthy",
  "service": "RAG System",
  "version": "1.0",
  "llm_model": "llama3.2"
}
```

**Example (curl):**
```bash
curl http://localhost:5001/health
```

---

## Error Responses

All endpoints return appropriate HTTP status codes and error messages:

**400 Bad Request:**
```json
{
  "error": "No file provided"
}
```

**404 Not Found:**
```json
{
  "error": "Endpoint not found"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Internal server error details..."
}
```

---

## Usage Examples

### Complete Workflow: Document Q&A

```bash
# 1. Upload a document
curl -X POST http://localhost:5001/upload/file -F "file=@report.pdf"

# 2. Create a chat session
SESSION_ID=$(curl -X POST http://localhost:5001/chat/new | jq -r '.session_id')

# 3. Ask questions
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is the main conclusion?\", \"session_id\": \"$SESSION_ID\"}"

# 4. Get history
curl http://localhost:5001/chat/history/$SESSION_ID
```

---

### Complete Workflow: Code Review

```bash
# 1. Upload code file
curl -X POST http://localhost:5001/upload/code -F "file=@app.py"

# 2. Quick review
curl -X POST http://localhost:5001/review/quick -F "file=@app.py"

# 3. Security review
curl -X POST http://localhost:5001/review/security -F "file=@app.py"

# 4. Get improvement suggestions
curl -X POST http://localhost:5001/review/improve \
  -F "file=@app.py" \
  -F "goal=Improve error handling"

# 5. Ask specific questions via chat
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How can I optimize the database queries in app.py?"}'
```

---

## Notes

- All file uploads support multipart/form-data
- JSON requests support application/json content type
- Session IDs are generated automatically if not provided
- The system maintains context within sessions
- Code and documents share the same vector database
- Context from related code/documents is automatically included in reviews
