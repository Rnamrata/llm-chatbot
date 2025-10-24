# How The System Works - Complete Explanation

## Table of Contents
1. [System Overview](#system-overview)
2. [RAG (Retrieval-Augmented Generation) Explained](#rag-explained)
3. [Vector Embeddings Explained](#vector-embeddings-explained)
4. [Complete Request Lifecycle](#complete-request-lifecycle)
5. [Code Review Process](#code-review-process)
6. [Session Management](#session-management)
7. [Behind The Scenes](#behind-the-scenes)

---

## System Overview

### What This System Does

This is a **smart document and code analysis system** that can:

1. **Understand your documents**: Upload PDFs, text files, or web pages
2. **Answer questions**: Ask questions about uploaded content
3. **Review code**: Get AI-powered code analysis and suggestions
4. **Remember context**: Maintain conversation history
5. **Find relevant information**: Use semantic search to find related content

### How It's Different From ChatGPT

| Feature | ChatGPT | This System |
|---------|---------|-------------|
| **Knowledge** | Fixed training data | Your uploaded documents |
| **Privacy** | Cloud-based | 100% local |
| **Code Review** | General feedback | Language-specific analysis |
| **Context** | Limited | Full codebase awareness |
| **Customization** | Limited | Fully customizable |

---

## RAG (Retrieval-Augmented Generation) Explained

### The Problem RAG Solves

**Normal LLM**:
- Only knows what it was trained on
- Can't access your specific documents
- May hallucinate or make up information

**RAG System**:
- Combines LLM with your documents
- Retrieves relevant information first
- Generates answers based on actual content

### How RAG Works - Step by Step

```
┌─────────────────────────────────────────────┐
│ Step 1: User Uploads Document               │
│                                             │
│ "How to configure authentication in my app" │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ Step 2: Document Processing                 │
│                                             │
│ • Extract text from PDF/file                │
│ • Split into chunks (1000 chars each)       │
│ • Add metadata (source, page, etc.)         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ Step 3: Generate Embeddings                 │
│                                             │
│ Text: "Authentication setup requires..."    │
│ ↓                                           │
│ Vector: [0.123, -0.456, 0.789, ...]        │
│         (768 numbers representing meaning)  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ Step 4: Store in Vector Database            │
│                                             │
│ ChromaDB saves embeddings + original text   │
└─────────────────────────────────────────────┘

When user asks a question:

┌─────────────────────────────────────────────┐
│ Step 5: User Asks Question                  │
│                                             │
│ "How do I set up authentication?"           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ Step 6: Convert Question to Embedding       │
│                                             │
│ Question → [0.145, -0.423, 0.891, ...]     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ Step 7: Search Vector Database              │
│                                             │
│ Find chunks with similar embeddings         │
│ (cosine similarity > 0.75)                  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ Step 8: Retrieve Top 5 Most Relevant Chunks │
│                                             │
│ 1. "Authentication setup requires..."       │
│ 2. "Configure JWT tokens in..."             │
│ 3. "User login endpoint..."                 │
│ 4. "Password hashing with bcrypt..."        │
│ 5. "Session management..."                  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ Step 9: Build Prompt for LLM                │
│                                             │
│ "Based on these documents:                  │
│  [Retrieved chunks]                         │
│                                             │
│  Answer this question:                      │
│  'How do I set up authentication?'"         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ Step 10: LLM Generates Answer               │
│                                             │
│ Ollama (llama3.2) processes the prompt      │
│ and generates answer based on retrieved docs│
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ Step 11: Return Answer + Sources            │
│                                             │
│ Answer: "To set up authentication..."       │
│ Sources: [chunk1, chunk2, chunk3]           │
└─────────────────────────────────────────────┘
```

### Key RAG Components

1. **Chunking**: Breaking documents into manageable pieces
2. **Embedding**: Converting text to numerical vectors
3. **Indexing**: Storing vectors for fast search
4. **Retrieval**: Finding similar content
5. **Augmentation**: Adding retrieved content to prompt
6. **Generation**: LLM creates answer

---

## Vector Embeddings Explained

### What Are Embeddings?

Embeddings are **numerical representations of text meaning**.

**Example**:
```
Text: "The cat sat on the mat"
Embedding: [0.23, -0.45, 0.78, 0.12, -0.34, ...] (768 numbers)

Text: "A feline rested on the rug"
Embedding: [0.25, -0.43, 0.76, 0.15, -0.32, ...] (similar numbers!)
```

Even though the words are different, the **meanings are similar**, so the embeddings are close.

### How Similarity Works

**Cosine Similarity**:
```
Vector A: [1, 2, 3]
Vector B: [2, 4, 6]

Similarity = (A • B) / (|A| × |B|)
            = (1×2 + 2×4 + 3×6) / (√14 × √56)
            = 0.997 (very similar!)

Vector C: [-1, -2, -3]
Similarity with A = -0.997 (opposite meaning)
```

**In our system**:
- Similarity > 0.75 = Very relevant
- Similarity 0.5-0.75 = Somewhat relevant
- Similarity < 0.5 = Not relevant

### Why Use Embeddings?

**Traditional Keyword Search**:
```
Query: "authentication"
Finds: Documents containing exactly "authentication"
Misses: Documents with "login", "auth", "sign in"
```

**Embedding Search**:
```
Query: "authentication"
Embedding: [0.34, -0.56, ...]
Finds:
  - "authentication" (similarity: 0.95)
  - "user login" (similarity: 0.82)
  - "access control" (similarity: 0.76)
  - "sign in process" (similarity: 0.74)
```

Embeddings understand **meaning**, not just words!

---

## Complete Request Lifecycle

### Example: Uploading and Querying a Document

#### 1. User Uploads PDF Document

**HTTP Request**:
```http
POST /upload/file HTTP/1.1
Content-Type: multipart/form-data

file: authentication-guide.pdf
```

**Server Processing**:

**Step 1**: Multer intercepts request
```javascript
// Multer middleware runs first
upload.single('file')  // Stores file in memory as buffer
```

**Step 2**: Route handler receives file
```javascript
app.post('/upload/file', upload.single('file'), async (req, res) => {
    // req.file contains:
    // {
    //   buffer: <PDF binary data>,
    //   originalname: "authentication-guide.pdf",
    //   mimetype: "application/pdf"
    // }

    const result = await fileManager.uploadFile(req.file);
    res.json(result);
});
```

**Step 3**: FileManager processes
```javascript
async uploadFile(file) {
    // 1. Save file to disk
    await fs.writeFile('uploads/authentication-guide.pdf', file.buffer);

    // 2. Detect type - it's a PDF
    if (filename.endsWith('.pdf')) {
        // 3. Extract text
        content = await documentProcessor.pdfToText(path, filename);
        // content = "Authentication in modern web applications..."
    }

    // 4. Chunk the content
    chunks = await documentProcessor.chunkDocument(content, {
        source: 'authentication-guide.pdf',
        type: 'document'
    });
    // chunks = [
    //   { pageContent: "Authentication in modern...", metadata: {...} },
    //   { pageContent: "JWT tokens provide...", metadata: {...} },
    //   ...
    // ]

    // 5. Store in vector database
    await vectorStore.storeChunks(chunks);

    return { success: true, chunks_created: 15 };
}
```

**Step 4**: DocumentProcessor chunks text
```javascript
async chunkDocument(content, metadata) {
    // 1. Create Document object
    const doc = new Document({
        pageContent: content,  // Full PDF text
        metadata: { source: 'authentication-guide.pdf', type: 'document' }
    });

    // 2. Apply markdown splitting (respects headers)
    const mdSplits = await markdownSplitter.splitDocuments([doc]);
    // Splits at "# Header" and "## Subheader"

    // 3. Apply recursive splitting
    const finalChunks = await recursiveSplitter.splitDocuments(mdSplits);
    // Tries separators: "\n\n\n", "\n\n", "\n", ".", " "
    // Until chunk is <= 1000 characters

    // Result:
    // [
    //   { pageContent: "Authentication in modern web applications
    //                   requires careful consideration of security...",
    //     metadata: {...} },
    //   { pageContent: "JWT tokens provide a stateless way to manage
    //                   user sessions. Here's how they work...",
    //     metadata: {...} }
    // ]

    return finalChunks;
}
```

**Step 5**: VectorStore generates embeddings
```javascript
async storeChunks(chunks) {
    // For each chunk:
    for (const chunk of chunks) {
        // 1. Send text to Ollama embedding model
        const embedding = await ollamaEmbeddings.embedQuery(chunk.pageContent);
        // HTTP POST to http://localhost:11434/api/embeddings
        // {
        //   "model": "nomic-embed-text",
        //   "prompt": "Authentication in modern web applications..."
        // }
        // Response: { embedding: [0.234, -0.567, 0.891, ..., ] }  // 768 numbers

        // 2. Store in ChromaDB
        await chromaDB.add({
            ids: [uuid()],
            embeddings: [embedding],
            metadatas: [chunk.metadata],
            documents: [chunk.pageContent]
        });
    }
}
```

**Response to Client**:
```json
{
    "success": true,
    "message": "File uploaded and processed successfully",
    "filename": "authentication-guide.pdf",
    "file_type": "document",
    "chunks_created": 15
}
```

---

#### 2. User Asks Question

**HTTP Request**:
```http
POST /chat HTTP/1.1
Content-Type: application/json

{
    "question": "How do JWT tokens work?",
    "session_id": "abc-123"
}
```

**Server Processing**:

**Step 1**: ChatSession receives question
```javascript
async query(question, sessionId) {
    // 1. Get or create session
    if (!sessions.has(sessionId)) {
        await createSession(sessionId);
        // Creates:
        // - Retriever (for vector search)
        // - ConversationalChain (LangChain)
        // - Memory (for chat history)
    }

    const session = sessions.get(sessionId);

    // 2. Run the conversational chain
    const result = await session.chain.call({ question });

    return result;
}
```

**Step 2**: Conversational Chain executes
```javascript
// Internal LangChain process:

// 2a. Embed the question
const questionEmbedding = await embeddings.embedQuery("How do JWT tokens work?");
// [0.145, -0.523, 0.876, ...]

// 2b. Search vector database
const docs = await vectorstore.similaritySearch(questionEmbedding, k=5);
// Returns 5 most similar chunks:
// [
//   { pageContent: "JWT tokens provide a stateless way...",
//     similarity: 0.89 },
//   { pageContent: "JSON Web Tokens consist of three parts...",
//     similarity: 0.85 },
//   { pageContent: "The signature ensures token integrity...",
//     similarity: 0.78 },
//   ...
// ]

// 2c. Build prompt
const prompt = `
Use the following pieces of context to answer the question.

Context:
${docs.map(d => d.pageContent).join('\n\n')}

Chat History:
${session.memory.chatHistory}

Question: ${question}

Answer:`;

// 2d. Send to LLM
const answer = await ollama.call(prompt);
// HTTP POST to http://localhost:11434/api/generate
// {
//   "model": "llama3.2",
//   "prompt": "Use the following pieces of context...",
//   "stream": false
// }

// 2e. LLM generates answer
// "JWT tokens work by encoding user information in three parts:
//  header, payload, and signature. The header specifies the algorithm,
//  the payload contains claims, and the signature ensures integrity..."

return { answer, sourceDocuments: docs };
```

**Step 3**: Format and return response
```javascript
// Format sources
const sources = docs.map(doc => ({
    content: doc.pageContent.substring(0, 200) + '...',
    metadata: doc.metadata,
    full_length: doc.pageContent.length
}));

return {
    success: true,
    answer: "JWT tokens work by encoding user information...",
    sources: sources,
    num_sources: 5,
    session_id: sessionId,
    message_count: 1
};
```

**Response to Client**:
```json
{
    "success": true,
    "answer": "JWT tokens work by encoding user information in three parts: header, payload, and signature. The header specifies the algorithm, the payload contains claims, and the signature ensures integrity using a secret key. This allows stateless authentication without server-side session storage.",
    "sources": [
        {
            "content": "JWT tokens provide a stateless way to manage user sessions. Here's how they work...",
            "metadata": { "source": "authentication-guide.pdf", "type": "document" },
            "full_length": 542
        }
    ],
    "num_sources": 5,
    "session_id": "abc-123",
    "message_count": 1
}
```

---

## Code Review Process

### Example: Reviewing JavaScript Code

**HTTP Request**:
```http
POST /review/comprehensive HTTP/1.1
Content-Type: multipart/form-data

file: login.js
question: Is this code secure?
```

**File Content** (`login.js`):
```javascript
function login(username, password) {
    const user = db.query("SELECT * FROM users WHERE username = '" + username + "'");
    if (user && user.password === password) {
        return { token: generateToken(user) };
    }
    return null;
}
```

**Server Processing**:

**Step 1**: Extract code and detect language
```javascript
const filename = 'login.js';
const codeContent = file.buffer.toString('utf-8');

const parser = new CodeParser();
const language = parser.detectLanguage(filename);
// Looks at extension: '.js' → 'javascript'
```

**Step 2**: Search for related code
```javascript
const contextDocs = await vectorStore.search(
    codeContent.substring(0, 500),  // First 500 chars
    k=3
);
// Finds similar code that was previously uploaded:
// - Previous authentication functions
// - Database query examples
// - Security implementations
```

**Step 3**: Format context
```javascript
const context = formatCodeContext(contextDocs);
// Result:
// "
// **From auth.js (lines 23-45):**
// ```javascript
// function secureLogin(username, password) {
//     const stmt = db.prepare('SELECT * FROM users WHERE username = ?');
//     const user = stmt.get(username);
//     if (user && await bcrypt.compare(password, user.hash)) {
//         return { token: jwt.sign({ id: user.id }, SECRET) };
//     }
//     return null;
// }
// ```
// "
```

**Step 4**: Select and format prompt
```javascript
const promptTemplate = getReviewPromptTemplate('comprehensive');
const prompt = await promptTemplate.format({
    code: codeContent,
    language: 'javascript',
    source: 'login.js',
    context: context,
    chat_history: '',
    question: 'Is this code secure?'
});

// Full prompt sent to LLM:
// "You are an expert code reviewer with years of experience in software
//  engineering best practices, security, and performance optimization.
//
//  Review the following code and provide detailed, actionable feedback:
//
//  **Code to Review:**
//  ```javascript
//  function login(username, password) {
//      const user = db.query("SELECT * FROM users WHERE username = '" + username + "'");
//      if (user && user.password === password) {
//          return { token: generateToken(user) };
//      }
//      return null;
//  }
//  ```
//
//  **Source:** login.js
//  **Context from codebase:**
//  [Related secure auth code...]
//
//  **Specific Question:** Is this code secure?
//
//  Please analyze the code for:
//  1. Security Issues
//  2. Best Practices
//  ..."
```

**Step 5**: LLM analyzes and responds
```javascript
const review = await llm.call(prompt);

// LLM Response:
// "CRITICAL SECURITY ISSUES FOUND:
//
//  1. **SQL Injection Vulnerability** (CRITICAL)
//     Line 2: Direct string concatenation in SQL query
//
//     Current code:
//     db.query(\"SELECT * FROM users WHERE username = '\" + username + \"'\")
//
//     Attack example:
//     username = \"admin' OR '1'='1\"
//     Query becomes: SELECT * FROM users WHERE username = 'admin' OR '1'='1'
//     This returns all users!
//
//     Fix:
//     const stmt = db.prepare('SELECT * FROM users WHERE username = ?');
//     const user = stmt.get(username);
//
//  2. **Plain Text Password Comparison** (CRITICAL)
//     Line 3: Comparing passwords directly
//
//     Passwords should be hashed with bcrypt:
//     if (user && await bcrypt.compare(password, user.passwordHash)) {
//         ...
//     }
//
//  3. **Missing Input Validation** (HIGH)
//     No validation on username/password inputs
//
//     Add:
//     if (!username || !password) return null;
//     if (username.length > 255) return null;
//
//  RECOMMENDATIONS:
//  - Use prepared statements
//  - Hash passwords with bcrypt
//  - Add input validation
//  - Implement rate limiting
//  - Use secure token generation
//  ..."
```

**Response to Client**:
```json
{
    "success": true,
    "filename": "login.js",
    "language": "javascript",
    "review_type": "comprehensive",
    "review": "CRITICAL SECURITY ISSUES FOUND:\n\n1. **SQL Injection Vulnerability** (CRITICAL)...",
    "context_used": 3
}
```

---

## Session Management

### How Sessions Work

**Session Lifecycle**:

```
1. Client sends first message
   ↓
2. Server checks if session exists
   ↓
3. If not, create new session:
   - Generate UUID
   - Create conversational chain
   - Initialize memory
   - Store in sessions Map
   ↓
4. Process message with session's chain
   ↓
5. Chain uses session's memory for context
   ↓
6. Update last_activity timestamp
   ↓
7. Increment message_count
   ↓
8. Return response
```

**Session Data Structure**:
```javascript
{
    sessionId: "550e8400-e29b-41d4-a716-446655440000",
    chain: ConversationalRetrievalQAChain {
        llm: Ollama,
        retriever: VectorStoreRetriever,
        memory: BufferMemory
    },
    memory: BufferMemory {
        chatHistory: {
            messages: [
                HumanMessage("How do JWT tokens work?"),
                AIMessage("JWT tokens work by encoding..."),
                HumanMessage("Can you show an example?"),
                AIMessage("Sure! Here's an example...")
            ]
        }
    },
    created_at: "2025-10-21T10:30:00Z",
    last_activity: "2025-10-21T10:35:00Z",
    message_count: 4
}
```

**Memory in Action**:

```
Message 1:
User: "How do JWT tokens work?"
Memory: []
LLM: "JWT tokens work by encoding..."

Message 2:
User: "Can you show an example?"
Memory: [
    "Q: How do JWT tokens work?",
    "A: JWT tokens work by encoding..."
]
LLM: "Based on our previous discussion about JWT tokens,
      here's an example..."  ← References previous context!

Message 3:
User: "What about the signature part?"
Memory: [
    "Q: How do JWT tokens work?",
    "A: JWT tokens work by encoding...",
    "Q: Can you show an example?",
    "A: Based on our previous discussion..."
]
LLM: "The signature part, as I mentioned in the example,
      ensures token integrity..."  ← Knows full conversation!
```

**Session Cleanup**:

```javascript
// Automatic cleanup of inactive sessions
cleanupInactiveSessions(inactiveHours = 24) {
    const now = new Date();
    const cutoff = now - (24 * 60 * 60 * 1000);  // 24 hours ago

    for (const [sessionId, session] of sessions.entries()) {
        if (session.last_activity < cutoff) {
            sessions.delete(sessionId);  // Remove old session
        }
    }
}
```

---

## Behind The Scenes

### What Happens When You Start The Server

```
1. Node.js loads server.js
   ↓
2. Import all modules
   ├─ fileManager.js
   ├─ llmManager.js
   ├─ chatSession.js
   └─ ... others
   ↓
3. Initialize instances
   ├─ documentProcessor = new DocumentProcessor()
   │   └─ Creates text splitters (markdown, recursive)
   │
   ├─ vectorStore = new VectorStoreAndEmbedding()
   │   ├─ Initialize Ollama embeddings
   │   └─ Connect to ChromaDB
   │       └─ HTTP connection to localhost:8000
   │
   ├─ fileManager = new FileManager(documentProcessor, vectorStore)
   │
   ├─ llmManager = new LLMManager('llama3.2', 0.7)
   │   └─ Create Ollama LLM instance
   │       └─ HTTP connection to localhost:11434
   │
   └─ chatManager = new ChatSession(llmManager, vectorStore)
       └─ Initialize empty sessions Map
   ↓
4. Set up Express middleware
   ├─ CORS (cross-origin requests)
   ├─ JSON parser
   ├─ URL-encoded parser
   └─ Multer (file uploads)
   ↓
5. Register all routes
   ├─ /upload/* routes
   ├─ /review/* routes
   ├─ /chat/* routes
   └─ /health, /stats
   ↓
6. Start HTTP server on port 5001
   ↓
7. Server is ready!
   Console: "🌐 Server starting on http://0.0.0.0:5001"
```

### What Happens When The Server Receives A Request

```
1. HTTP request arrives at server
   ↓
2. Express routing layer
   ├─ Match request path to route
   ├─ Check HTTP method (GET/POST/DELETE)
   └─ Execute middleware chain
   ↓
3. Middleware execution (in order)
   ├─ CORS: Add access control headers
   ├─ JSON parser: Parse request body if JSON
   ├─ Multer: Handle file upload if present
   └─ Route-specific middleware
   ↓
4. Route handler execution
   ├─ Extract parameters (body, params, query)
   ├─ Validate inputs
   ├─ Call appropriate module method
   ├─ await result
   ├─ Format response
   └─ Send JSON response
   ↓
5. Error handling (if error occurs)
   ├─ Catch in try-catch
   ├─ Log error
   ├─ Send error response with status code
   ↓
6. Response sent to client
   └─ Connection closed
```

### File System Operations

**Upload Directory Structure**:
```
uploads/
├── authentication-guide.pdf
├── user-manual.txt
├── code-review-example.js
└── media/
    ├── youtube-video-123.mp3
    ├── youtube-video-123_transcription.txt
    └── youtube-video-456.mp3
```

**ChromaDB Storage**:
```
chroma_db/
└── my_documents/
    ├── index/              # HNSW index for fast vector search
    ├── data/               # Document content
    ├── embeddings/         # Vector embeddings
    └── metadata/           # Document metadata
```

---

This comprehensive explanation covers how every part of the system works together to provide intelligent document analysis and code review capabilities!
