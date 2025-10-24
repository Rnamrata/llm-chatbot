# Complete API Documentation

## Base URL
```
http://localhost:5001
```

## Table of Contents
1. [Upload Endpoints](#upload-endpoints)
2. [Code Review Endpoints](#code-review-endpoints)
3. [Chat Endpoints](#chat-endpoints)
4. [Utility Endpoints](#utility-endpoints)
5. [Error Responses](#error-responses)
6. [Request Examples](#request-examples)

---

## Upload Endpoints

### 1. Upload File (Document)

Upload PDF, TXT, or Markdown files for processing and storage.

**Endpoint**: `POST /upload/file`

**Content-Type**: `multipart/form-data`

**Request Body**:
```
file: <binary file data>
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "message": "File uploaded and processed successfully",
    "filename": "document.pdf",
    "file_type": "document",
    "chunks_created": 15
}
```

**Error Response** (400 Bad Request):
```json
{
    "error": "Unsupported file type",
    "success": false
}
```

**Curl Example**:
```bash
curl -X POST http://localhost:5001/upload/file \
  -F "file=@document.pdf"
```

---

### 2. Upload Code File

Upload code file with complexity analysis.

**Endpoint**: `POST /upload/code`

**Content-Type**: `multipart/form-data`

**Request Body**:
```
file: <code file>
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "message": "Code file uploaded for review successfully",
    "filename": "auth.js",
    "language": "javascript",
    "chunks_created": 8,
    "complexity": {
        "lines_of_code": 234,
        "num_functions": 12,
        "num_classes": 2,
        "num_imports": 8,
        "cyclomatic_complexity": 45
    }
}
```

**Supported Languages**:
- Python (.py)
- JavaScript (.js, .jsx)
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
- HTML (.html)
- CSS (.css)
- SQL (.sql)
- Shell (.sh, .bash)

**Curl Example**:
```bash
curl -X POST http://localhost:5001/upload/code \
  -F "file=@authentication.js"
```

---

### 3. Upload YouTube Video

Process YouTube video (download + transcribe).

**Endpoint**: `POST /upload/youtube`

**Content-Type**: `application/json` or `application/x-www-form-urlencoded`

**Request Body**:
```json
{
    "url": "https://youtube.com/watch?v=VIDEO_ID"
}
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "message": "YouTube video processed and stored successfully",
    "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
    "transcription_file": "Rick_Astley_Never_Gonna_Give_You_Up_transcription.txt",
    "chunks_created": 42
}
```

**Process Flow**:
1. Download audio from YouTube
2. Transcribe using Whisper model
3. Save transcription as text file
4. Chunk transcription
5. Generate embeddings
6. Store in vector database

**Curl Example**:
```bash
curl -X POST http://localhost:5001/upload/youtube \
  -H "Content-Type: application/json" \
  -d '{"url":"https://youtube.com/watch?v=dQw4w9WgXcQ"}'
```

---

### 4. Upload Web Page

Scrape and process web page content.

**Endpoint**: `POST /upload/web`

**Content-Type**: `application/json`

**Request Body**:
```json
{
    "url": "https://example.com/article"
}
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "message": "Web page processed and stored successfully",
    "url": "https://example.com/article",
    "filename": "How_to_Build_a_Chatbot.txt",
    "chunks_created": 18
}
```

**Curl Example**:
```bash
curl -X POST http://localhost:5001/upload/web \
  -H "Content-Type: application/json" \
  -d '{"url":"https://docs.example.com/authentication"}'
```

---

## Code Review Endpoints

### 1. Quick Review

Fast review focusing on critical issues only.

**Endpoint**: `POST /review/quick`

**Content-Type**: `multipart/form-data`

**Request Body**:
```
file: <code file>
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "filename": "login.js",
    "language": "javascript",
    "review_type": "quick",
    "review": "CRITICAL ISSUES:\n\n1. SQL Injection on line 5...\n2. Missing input validation..."
}
```

**Review Focus**:
- Critical bugs
- Security vulnerabilities
- Major performance issues
- Obvious best practice violations

**Curl Example**:
```bash
curl -X POST http://localhost:5001/review/quick \
  -F "file=@login.js"
```

---

### 2. Comprehensive Review

Detailed analysis of all code aspects.

**Endpoint**: `POST /review/comprehensive`

**Content-Type**: `multipart/form-data` or `application/json`

**Request Body (File Upload)**:
```
file: <code file>
question: <optional specific focus>
```

**Request Body (JSON)**:
```json
{
    "code": "function example() { ... }",
    "filename": "example.js",
    "question": "Review for security and performance"
}
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "filename": "auth.js",
    "language": "javascript",
    "review_type": "comprehensive",
    "review": "CODE QUALITY ANALYSIS:\n\n## Readability\n- Function names are descriptive...\n\n## Security\n- Line 23: Use parameterized queries...",
    "context_used": 3
}
```

**Review Covers**:
1. Code Quality & Readability
2. Best Practices & Design Patterns
3. Potential Bugs
4. Security Issues
5. Performance Concerns
6. Testing & Maintainability

**Curl Example (File)**:
```bash
curl -X POST http://localhost:5001/review/comprehensive \
  -F "file=@authentication.js" \
  -F "question=Focus on security best practices"
```

**Curl Example (JSON)**:
```bash
curl -X POST http://localhost:5001/review/comprehensive \
  -H "Content-Type: application/json" \
  -d '{
    "code": "function login(user, pass) { return user.password === pass; }",
    "filename": "auth.js",
    "question": "Is this secure?"
  }'
```

---

### 3. Security Review

Security-focused analysis (OWASP Top 10).

**Endpoint**: `POST /review/security`

**Content-Type**: `multipart/form-data`

**Request Body**:
```
file: <code file>
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "filename": "api.js",
    "language": "javascript",
    "review_type": "security",
    "review": "SECURITY VULNERABILITIES FOUND:\n\n**CRITICAL**\n1. SQL Injection (Line 45)...\n\n**HIGH**\n2. Missing authentication check..."
}
```

**Security Checks**:
- Input validation
- SQL injection
- XSS vulnerabilities
- Authentication/authorization
- Sensitive data exposure
- Insecure dependencies
- Cryptography misuse
- OWASP Top 10

**Curl Example**:
```bash
curl -X POST http://localhost:5001/review/security \
  -F "file=@api-endpoints.js"
```

---

### 4. Performance Review

Performance optimization analysis.

**Endpoint**: `POST /review/performance`

**Content-Type**: `multipart/form-data`

**Request Body**:
```
file: <code file>
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "filename": "data-processor.js",
    "language": "javascript",
    "review_type": "performance",
    "review": "PERFORMANCE ANALYSIS:\n\n## Algorithm Complexity\n- Line 12: O(n²) loop, optimize to O(n)...\n\n## Memory Usage\n- Consider streaming for large datasets..."
}
```

**Performance Checks**:
- Algorithm complexity (Big O)
- Memory usage and leaks
- Database query optimization
- I/O operations efficiency
- Caching opportunities
- Concurrency issues
- Resource management

**Curl Example**:
```bash
curl -X POST http://localhost:5001/review/performance \
  -F "file=@data-processing.js"
```

---

### 5. Explain Code

Educational code explanation.

**Endpoint**: `POST /review/explain`

**Content-Type**: `multipart/form-data` or `application/json`

**Request Body (File)**:
```
file: <code file>
question: <optional specific question>
```

**Request Body (JSON)**:
```json
{
    "code": "const hash = await bcrypt.hash(password, 10);",
    "filename": "auth.js",
    "question": "What does the second parameter mean?"
}
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "filename": "crypto.js",
    "language": "javascript",
    "explanation": "This code implements password hashing using bcrypt...\n\nStep by step:\n1. bcrypt.hash() takes two parameters...\n2. The number 10 is the salt rounds..."
}
```

**Curl Example**:
```bash
curl -X POST http://localhost:5001/review/explain \
  -H "Content-Type: application/json" \
  -d '{
    "code": "app.use(cors())",
    "filename": "server.js",
    "question": "What does cors() do?"
  }'
```

---

### 6. Detect Bugs

Bug detection and analysis.

**Endpoint**: `POST /review/bugs`

**Content-Type**: `multipart/form-data` or `application/json`

**Request Body**:
```json
{
    "code": "function divide(a, b) { return a / b; }",
    "filename": "math.js",
    "issue": "Sometimes returns Infinity"
}
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "filename": "math.js",
    "language": "javascript",
    "bug_analysis": "BUGS FOUND:\n\n1. **Division by Zero** (Line 1)\n   - When b is 0, returns Infinity\n   - Fix: Add validation...\n   - Test case: divide(5, 0) should throw error"
}
```

**Bug Types Detected**:
- Logic errors
- Edge cases not handled
- Race conditions
- Off-by-one errors
- Null/undefined issues
- Type mismatches
- Error handling gaps

**Curl Example**:
```bash
curl -X POST http://localhost:5001/review/bugs \
  -F "file=@calculator.js" \
  -F "issue=Returns NaN for some inputs"
```

---

### 7. Suggest Improvements

Code improvement and refactoring suggestions.

**Endpoint**: `POST /review/improve`

**Content-Type**: `multipart/form-data` or `application/json`

**Request Body**:
```json
{
    "code": "if (user.role === 'admin') { return true; } else { return false; }",
    "filename": "auth.js",
    "goal": "Make it more concise"
}
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "filename": "auth.js",
    "language": "javascript",
    "suggestions": "IMPROVEMENT SUGGESTIONS:\n\n1. Simplify boolean return:\n   Current: if (condition) { return true; } else { return false; }\n   Better: return condition\n\n   Updated code:\n   return user.role === 'admin';"
}
```

**Curl Example**:
```bash
curl -X POST http://localhost:5001/review/improve \
  -F "file=@legacy-code.js" \
  -F "goal=Modernize to ES6+ syntax"
```

---

## Chat Endpoints

### 1. Send Chat Message

Ask questions about uploaded documents.

**Endpoint**: `POST /chat`

**Content-Type**: `application/json`

**Request Body**:
```json
{
    "question": "How do I implement JWT authentication?",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "k": 5
}
```

**Parameters**:
- `question` (required): Your question
- `session_id` (optional): Session ID for conversation continuity
- `k` (optional): Number of relevant chunks to retrieve (default: 5)

**Success Response** (200 OK):
```json
{
    "success": true,
    "answer": "To implement JWT authentication, you need to...\n\nBased on the uploaded documentation, here are the steps:\n1. Install jsonwebtoken package\n2. Create a secret key...",
    "sources": [
        {
            "content": "JWT authentication requires three main components: header, payload, and signature...",
            "metadata": {
                "source": "auth-guide.pdf",
                "type": "document",
                "chunk_index": 5
            },
            "full_length": 542
        }
    ],
    "num_sources": 5,
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message_count": 1
}
```

**How It Works**:
1. Embed question as vector
2. Search vector database for similar content
3. Retrieve top k most relevant chunks
4. Combine chunks with question
5. Send to LLM for answer generation
6. Return answer with source attribution

**Curl Example**:
```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the security best practices?",
    "k": 5
  }'
```

---

### 2. Create New Session

Create a new chat session.

**Endpoint**: `POST /chat/new`

**Content-Type**: `application/json`

**Request Body**: None (empty object)

**Success Response** (200 OK):
```json
{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "New chat session created"
}
```

**Curl Example**:
```bash
curl -X POST http://localhost:5001/chat/new
```

---

### 3. Get Session Info

Get information about a specific session.

**Endpoint**: `GET /chat/session/:sessionId`

**Parameters**:
- `sessionId`: Session ID

**Success Response** (200 OK):
```json
{
    "exists": true,
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-10-21T10:30:00.000Z",
    "last_activity": "2025-10-21T10:35:45.000Z",
    "message_count": 5
}
```

**Not Found Response** (200 OK):
```json
{
    "exists": false,
    "message": "Session not found"
}
```

**Curl Example**:
```bash
curl http://localhost:5001/chat/session/550e8400-e29b-41d4-a716-446655440000
```

---

### 4. Get Chat History

Retrieve conversation history for a session.

**Endpoint**: `GET /chat/history/:sessionId`

**Parameters**:
- `sessionId`: Session ID

**Success Response** (200 OK):
```json
{
    "history": [
        {
            "question": "How does JWT work?",
            "answer": "JWT (JSON Web Token) is a compact way to securely transmit information...",
            "timestamp": "2025-10-21T10:30:00.000Z"
        },
        {
            "question": "Can you show an example?",
            "answer": "Sure! Here's an example of JWT implementation...",
            "timestamp": "2025-10-21T10:30:00.000Z"
        }
    ],
    "length": 2,
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message_count": 2
}
```

**Curl Example**:
```bash
curl http://localhost:5001/chat/history/550e8400-e29b-41d4-a716-446655440000
```

---

### 5. Clear Session History

Delete a session and its history.

**Endpoint**: `DELETE /chat/clear/:sessionId`

**Parameters**:
- `sessionId`: Session ID

**Success Response** (200 OK):
```json
{
    "success": true,
    "message": "Conversation history cleared",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Curl Example**:
```bash
curl -X DELETE http://localhost:5001/chat/clear/550e8400-e29b-41d4-a716-446655440000
```

---

### 6. List All Sessions

Get list of all active sessions.

**Endpoint**: `GET /chat/sessions`

**Success Response** (200 OK):
```json
{
    "sessions": [
        {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "created_at": "2025-10-21T10:00:00.000Z",
            "last_activity": "2025-10-21T10:35:00.000Z",
            "message_count": 5
        },
        {
            "session_id": "660f9511-f39c-52e5-b827-557766551111",
            "created_at": "2025-10-21T11:00:00.000Z",
            "last_activity": "2025-10-21T11:15:00.000Z",
            "message_count": 3
        }
    ],
    "total_sessions": 2
}
```

**Curl Example**:
```bash
curl http://localhost:5001/chat/sessions
```

---

### 7. Cleanup Inactive Sessions

Remove sessions inactive for specified hours.

**Endpoint**: `POST /chat/cleanup`

**Content-Type**: `application/json`

**Request Body**:
```json
{
    "inactive_hours": 24
}
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "cleaned": 3,
    "remaining": 2,
    "message": "Cleaned up 3 inactive sessions"
}
```

**Curl Example**:
```bash
curl -X POST http://localhost:5001/chat/cleanup \
  -H "Content-Type: application/json" \
  -d '{"inactive_hours": 12}'
```

---

## Utility Endpoints

### 1. Get Statistics

Get database and system statistics.

**Endpoint**: `GET /stats`

**Success Response** (200 OK):
```json
{
    "total_chunks": 1523,
    "total_sessions": 5,
    "status": "ready",
    "message": "Vector database contains 1523 chunks"
}
```

**Curl Example**:
```bash
curl http://localhost:5001/stats
```

---

### 2. Health Check

Check if server is running.

**Endpoint**: `GET /health`

**Success Response** (200 OK):
```json
{
    "status": "healthy",
    "service": "RAG System",
    "version": "1.0",
    "llm_model": "llama3.2"
}
```

**Curl Example**:
```bash
curl http://localhost:5001/health
```

---

## Error Responses

### 400 Bad Request
```json
{
    "error": "No file provided"
}
```

### 404 Not Found
```json
{
    "error": "Endpoint not found"
}
```

### 500 Internal Server Error
```json
{
    "error": "Failed to process document"
}
```

---

## Request Examples

### Python Requests

```python
import requests

# Upload file
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:5001/upload/file',
        files={'file': f}
    )
print(response.json())

# Ask question
response = requests.post(
    'http://localhost:5001/chat',
    json={
        'question': 'How does authentication work?',
        'k': 5
    }
)
print(response.json()['answer'])

# Review code
with open('auth.js', 'rb') as f:
    response = requests.post(
        'http://localhost:5001/review/comprehensive',
        files={'file': f},
        data={'question': 'Is this secure?'}
    )
print(response.json()['review'])
```

### JavaScript Fetch

```javascript
// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:5001/upload/file', {
    method: 'POST',
    body: formData
});
const result = await response.json();

// Ask question
const response = await fetch('http://localhost:5001/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        question: 'How does JWT work?',
        session_id: 'my-session-id'
    })
});
const result = await response.json();
console.log(result.answer);
```

---

## Rate Limiting

Currently no rate limiting is implemented. For production:
- Implement per-IP rate limiting
- Add API key authentication
- Limit concurrent uploads
- Set maximum file sizes

## Authentication

Currently no authentication is required. For production:
- Add JWT-based authentication
- Implement API keys
- Role-based access control
- Session management with secure tokens
