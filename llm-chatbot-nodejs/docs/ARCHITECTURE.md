# System Architecture Documentation

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [System Components](#system-components)
3. [Data Flow](#data-flow)
4. [Technology Stack](#technology-stack)
5. [Module Interactions](#module-interactions)
6. [Database Architecture](#database-architecture)
7. [Security Architecture](#security-architecture)

---

## High-Level Overview

This system is a **RAG (Retrieval-Augmented Generation)** chatbot with code review capabilities. It combines:

- **Document Processing**: Extracts and chunks content from various sources
- **Vector Storage**: Stores embeddings for semantic search
- **LLM Integration**: Uses local Ollama models for privacy
- **Code Analysis**: Language-aware parsing and review
- **Session Management**: Maintains conversation context

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  (HTTP Requests: File Uploads, Questions, Code Review Requests) │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXPRESS.JS SERVER (Port 5001)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              REST API ENDPOINTS                           │  │
│  │  - /upload/*    - /review/*    - /chat/*                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌────────────────┐ ┌──────────────┐ ┌────────────────┐
│ FILE MANAGER   │ │ LLM MANAGER  │ │ CHAT SESSION   │
│   MODULE       │ │   MODULE     │ │   MANAGER      │
└───────┬────────┘ └──────┬───────┘ └────────┬───────┘
        │                 │                  │
        ▼                 ▼                  ▼
┌───────────────────────────────────────────────────────┐
│              SUPPORTING MODULES                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   Document   │  │     Code     │  │   Review    │ │
│  │  Processor   │  │    Parser    │  │   Prompts   │ │
│  └──────────────┘  └──────────────┘  └─────────────┘ │
└───────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌────────────────┐ ┌──────────────┐ ┌────────────────┐
│    CHROMADB    │ │   OLLAMA     │ │  FILE SYSTEM   │
│ Vector Storage │ │  LLM Engine  │ │   (uploads/)   │
│  (port 8000)   │ │ (port 11434) │ │                │
└────────────────┘ └──────────────┘ └────────────────┘
```

---

## System Components

### 1. Express.js Server (`server.js`)
**Role**: HTTP server and request router

**Responsibilities**:
- Accept incoming HTTP requests
- Route to appropriate handlers
- Apply middleware (CORS, multer, JSON parsing)
- Handle errors
- Send responses

**Key Features**:
- 20+ REST API endpoints
- File upload handling with multer
- JSON and form-data support
- Error handling middleware

---

### 2. File Manager Module (`fileManager.js`)
**Role**: Orchestrates file upload and processing pipeline

**Responsibilities**:
- Receive uploaded files
- Detect file type (document vs code)
- Route to appropriate processor
- Coordinate with document processor and code parser
- Store processed chunks in vector database

**Processing Pipeline**:
```
Upload → Type Detection → Processing → Chunking → Embedding → Storage
```

**Supported Types**:
- Documents: PDF, TXT, Markdown
- Code: 22+ programming languages
- Media: YouTube videos (download + transcribe)
- Web: HTML pages (scrape + extract)

---

### 3. Document Processor Module (`documentProcessor.js`)
**Role**: Extract and chunk document content

**Responsibilities**:
- Extract text from PDFs using pdf-parse
- Read text files
- Apply intelligent chunking strategies
- Preserve document structure
- Add metadata to chunks

**Chunking Strategy**:
1. **Markdown-aware splitting**: Respects headers
2. **Recursive splitting**: Uses multiple separators
3. **Fixed size chunks**: 1000 characters with 100 overlap
4. **Preserves context**: Overlapping prevents information loss

**Why Chunking?**
- LLMs have token limits
- Smaller chunks = more precise retrieval
- Overlap ensures context isn't lost at boundaries

---

### 4. Code Parser Module (`codeParser.js`)
**Role**: Language-aware code analysis and chunking

**Responsibilities**:
- Detect programming language from file extension
- Extract functions and classes
- Parse code structure
- Calculate complexity metrics
- Create semantic code chunks

**Language Support**:
- Python, JavaScript, TypeScript
- Java, Go, C/C++, Rust
- Ruby, PHP, Swift, Kotlin, C#
- HTML, CSS, SQL, Shell

**Parsing Strategies**:
- **Python**: Regex for `def` and `class` keywords
- **JavaScript**: Patterns for functions, arrow functions, classes
- **Generic**: Line-based chunking with size limits

**Complexity Metrics**:
- Lines of code
- Number of functions/classes
- Number of imports
- Cyclomatic complexity (decision points)

---

### 5. Vector Store Module (`vectorStoreAndEmbedding.js`)
**Role**: Semantic search and embedding storage

**Responsibilities**:
- Generate embeddings using Ollama
- Store embeddings in ChromaDB
- Perform similarity search
- Manage vector collection

**How Vector Search Works**:
```
1. User Query: "How do I authenticate users?"
   ↓
2. Generate Query Embedding (768-dimensional vector)
   ↓
3. Search ChromaDB for Similar Vectors (cosine similarity)
   ↓
4. Return Top K Most Similar Documents
   ↓
5. Send Documents + Query to LLM
```

**Embedding Model**: `nomic-embed-text`
- Fast and efficient
- 768 dimensions
- Trained for code and text

**Search Algorithm**: Maximum Marginal Relevance (MMR)
- Balances relevance and diversity
- Prevents redundant results

---

### 6. LLM Manager Module (`llmManager.js`)
**Role**: Interface to Ollama language models

**Responsibilities**:
- Initialize LLM connection
- Create conversational chains
- Format prompts
- Handle code review requests
- Process LLM responses

**Key Features**:
- **Multiple review types**: 10 different prompt templates
- **Context injection**: Adds related code for better analysis
- **Conversation memory**: Maintains chat history
- **Temperature control**: Configurable randomness

**LLM Chain Flow**:
```
Question → Format Prompt → Retrieve Context →
  → LLM Processing → Parse Response → Return Answer
```

---

### 7. Chat Session Manager (`chatSession.js`)
**Role**: Manage conversation sessions and history

**Responsibilities**:
- Create/destroy sessions
- Track conversation history
- Maintain session metadata
- Clean up inactive sessions

**Session Structure**:
```javascript
{
  sessionId: "uuid",
  chain: ConversationalRetrievalChain,
  memory: BufferMemory,
  created_at: Date,
  last_activity: Date,
  message_count: Number
}
```

**Memory Types**:
- **Buffer Memory**: Stores all messages in memory
- **Conversation History**: Maintains Q&A pairs
- **Context Window**: Sends recent history to LLM

---

### 8. Code Review Prompts (`codeReviewPrompts.js`)
**Role**: Specialized prompt templates for code analysis

**Template Types**:
1. **Comprehensive**: Full analysis (quality, security, performance)
2. **Quick**: Critical issues only
3. **Security**: OWASP Top 10, vulnerabilities
4. **Performance**: Algorithm complexity, optimization
5. **Best Practices**: Design patterns, SOLID
6. **Explanation**: Educational code walkthrough
7. **Bug Detection**: Finding potential errors
8. **Improvement**: Refactoring suggestions

**Template Variables**:
- `{code}`: Source code to review
- `{language}`: Programming language
- `{source}`: Filename
- `{context}`: Related code from codebase
- `{question}`: User's specific question

---

## Data Flow

### Upload Flow

```
1. Client uploads file via POST /upload/file
   ↓
2. Multer middleware intercepts, stores in memory
   ↓
3. Server passes to FileManager.uploadFile()
   ↓
4. FileManager detects type (PDF/Code/Text)
   ↓
5a. If PDF: DocumentProcessor.pdfToText()
5b. If Code: CodeParser.chunkCode()
5c. If Text: DocumentProcessor.textFileToText()
   ↓
6. DocumentProcessor.chunkDocument()
   - Applies markdown splitting
   - Applies recursive splitting
   - Creates Document objects with metadata
   ↓
7. VectorStore.storeChunks()
   - Generates embeddings via Ollama
   - Stores in ChromaDB
   ↓
8. Return success response with chunk count
```

### Chat Flow

```
1. Client sends question via POST /chat
   ↓
2. ChatSession receives question + session_id
   ↓
3. If new session: Create conversational chain
   ↓
4. VectorStore.search(question, k=5)
   - Embed question
   - Find similar documents
   ↓
5. LLM Chain combines:
   - Question
   - Retrieved documents
   - Conversation history
   ↓
6. Send to Ollama LLM
   ↓
7. LLM generates answer based on context
   ↓
8. Update session metadata
   ↓
9. Return answer + sources to client
```

### Code Review Flow

```
1. Client uploads code via POST /review/comprehensive
   ↓
2. Server extracts code and filename
   ↓
3. CodeParser.detectLanguage(filename)
   ↓
4. VectorStore.search(code, k=3)
   - Find similar code in database
   ↓
5. formatCodeContext(documents)
   - Format related code as markdown
   ↓
6. getReviewPromptTemplate('comprehensive')
   - Select appropriate prompt
   ↓
7. template.format({
     code, language, source, context, question
   })
   ↓
8. LLMManager.reviewCodeWithContext()
   - Send formatted prompt to Ollama
   ↓
9. LLM analyzes code and generates review
   ↓
10. Return structured review to client
```

---

## Technology Stack

### Backend Framework
- **Express.js**: Web server and routing
- **Node.js 18+**: JavaScript runtime

### LLM & Embeddings
- **Ollama**: Local LLM inference
- **llama3.2**: Language model
- **nomic-embed-text**: Embedding model

### Vector Database
- **ChromaDB**: Vector storage and similarity search

### Document Processing
- **pdf-parse**: PDF text extraction
- **Cheerio**: HTML parsing
- **LangChain.js**: RAG orchestration

### File Handling
- **Multer**: File upload middleware
- **ytdl-core**: YouTube download
- **@xenova/transformers**: Audio transcription

### Utilities
- **uuid**: Session ID generation
- **dotenv**: Environment configuration
- **cors**: Cross-origin requests

---

## Module Interactions

### Dependency Graph

```
server.js
├── fileManager.js
│   ├── documentProcessor.js
│   ├── codeParser.js
│   └── vectorStoreAndEmbedding.js
├── llmManager.js
│   └── codeReviewPrompts.js
├── chatSession.js
│   ├── llmManager.js
│   └── vectorStoreAndEmbedding.js
└── codeParser.js
```

### Communication Patterns

**File Manager → Document Processor**
- Synchronous method calls
- Passes file path and metadata
- Receives processed chunks

**File Manager → Vector Store**
- Async method calls
- Sends document chunks
- Awaits confirmation

**Chat Session → LLM Manager**
- Creates conversational chain
- Delegates LLM operations
- Receives formatted responses

**LLM Manager → Code Review Prompts**
- Imports prompt templates
- Selects appropriate template
- Formats with variables

---

## Database Architecture

### ChromaDB Collections

**Collection Name**: `my_documents`

**Document Structure**:
```javascript
{
  id: "auto-generated-uuid",
  embedding: [0.123, -0.456, ...],  // 768 dimensions
  metadata: {
    source: "filename.js",
    type: "code" | "document" | "youtube" | "web",
    content_type: "code" | "document",
    language: "javascript",
    structure_type: "function" | "class",
    structure_name: "myFunction",
    start_line: 10,
    end_line: 25,
    chunk_index: 0
  },
  document: "actual text content"
}
```

**Indexing Strategy**:
- **Vector Index**: HNSW (Hierarchical Navigable Small World)
- **Metadata Filters**: Can filter by language, type, source
- **Distance Metric**: Cosine similarity

---

## Security Architecture

### Input Validation
- File type checking
- File size limits (configurable)
- Code language validation
- URL validation for web scraping

### Data Privacy
- **Local LLM**: All processing on-premises
- **No cloud APIs**: No data sent to external services
- **Session isolation**: Each session independent

### File System Security
- Uploads stored in isolated directory
- Filename sanitization
- No executable permissions on uploads

### API Security
- CORS configuration
- Error message sanitization
- No sensitive data in logs

---

## Scalability Considerations

### Current Limitations
- **In-memory sessions**: Lost on restart
- **Single-threaded**: One request at a time per endpoint
- **Local storage**: Limited by disk space

### Scaling Strategies

**Horizontal Scaling**:
- Add Redis for session storage
- Use message queue for async processing
- Deploy multiple server instances with load balancer

**Vertical Scaling**:
- Increase ChromaDB memory
- Use faster embedding model
- Optimize chunk sizes

**Performance Optimization**:
- Cache frequently accessed embeddings
- Batch embedding generation
- Implement connection pooling

---

## Error Handling Architecture

### Error Types

1. **Client Errors (4xx)**
   - Missing files
   - Invalid file types
   - Malformed requests

2. **Server Errors (5xx)**
   - LLM connection failures
   - Database errors
   - Processing exceptions

### Error Flow

```
Error Occurs
   ↓
Catch in try-catch block
   ↓
Log error details
   ↓
Sanitize error message
   ↓
Return appropriate HTTP status
   ↓
Send JSON error response
```

### Logging Strategy
- Console logging for development
- Structured logs for production
- Error stack traces in dev only
- No sensitive data in logs

---

## Future Architecture Enhancements

### Planned Improvements

1. **Persistent Sessions**
   - Store in PostgreSQL or MongoDB
   - Session recovery on restart

2. **Caching Layer**
   - Redis for embeddings
   - Reduce redundant processing

3. **Async Processing**
   - Background job queue
   - Webhook notifications

4. **Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Health check endpoints

5. **Multi-tenancy**
   - User authentication
   - Isolated vector collections
   - Usage quotas

---

This architecture provides a solid foundation for a production-ready RAG system with room for growth and optimization.
