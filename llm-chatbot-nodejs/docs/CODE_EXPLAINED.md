# Code Explained - Line by Line Documentation

## Table of Contents
1. [Server.js Explained](#serverjs-explained)
2. [Vector Store Module Explained](#vector-store-module-explained)
3. [LLM Manager Module Explained](#llm-manager-module-explained)
4. [Chat Session Module Explained](#chat-session-module-explained)
5. [Code Parser Module Explained](#code-parser-module-explained)
6. [Document Processor Module Explained](#document-processor-module-explained)
7. [File Manager Module Explained](#file-manager-module-explained)

---

## Server.js Explained

### Imports and Setup

```javascript
import express from 'express';
```
- Imports the Express framework
- Express provides routing, middleware, and HTTP server capabilities
- Used to create REST API endpoints

```javascript
import cors from 'cors';
```
- Cross-Origin Resource Sharing middleware
- Allows requests from different domains (e.g., frontend on port 3000 calling API on port 5001)
- Without CORS, browsers block cross-origin requests

```javascript
import multer from 'multer';
```
- Middleware for handling `multipart/form-data` (file uploads)
- Stores uploaded files in memory as buffers
- Provides `req.file` and `req.files` to route handlers

```javascript
import { v4 as uuidv4 } from 'uuid';
```
- UUID (Universally Unique Identifier) generator
- Creates unique session IDs like "550e8400-e29b-41d4-a716-446655440000"
- Ensures session IDs never collide

```javascript
import dotenv from 'dotenv';
dotenv.config();
```
- Loads environment variables from `.env` file
- Makes `process.env.PORT`, `process.env.OLLAMA_URL`, etc. available
- Keeps configuration separate from code

### Express App Initialization

```javascript
const app = express();
const PORT = process.env.PORT || 5001;
```
- Creates Express application instance
- Sets port from environment or defaults to 5001
- `app` is the core object for routing and middleware

```javascript
app.use(cors());
```
- Enables CORS for all routes
- Adds headers: `Access-Control-Allow-Origin: *`
- Allows frontend applications to call this API

```javascript
app.use(express.json());
```
- Parses JSON request bodies
- Converts `{"question": "test"}` to `req.body.question`
- Automatically handles Content-Type: application/json

```javascript
app.use(express.urlencoded({ extended: true }));
```
- Parses URL-encoded form data
- Handles `Content-Type: application/x-www-form-urlencoded`
- `extended: true` allows rich objects and arrays

### Multer Configuration

```javascript
const upload = multer({ storage: multer.memoryStorage() });
```
**What this does**:
- Configures file upload handling
- `memoryStorage()` stores files in RAM (not disk)
- Uploaded files available as `req.file.buffer`

**Why memory storage?**:
- Faster than disk I/O
- Files are temporary (processed immediately)
- No cleanup needed (garbage collected)

**How it works**:
```javascript
// When client uploads file:
POST /upload/file
Content-Type: multipart/form-data

file=<binary data>

// Multer intercepts and creates:
req.file = {
    fieldname: 'file',
    originalname: 'document.pdf',
    encoding: '7bit',
    mimetype: 'application/pdf',
    buffer: <Buffer 25 50 44 46 2d 31 2e 34 0a 25 ...>,
    size: 52480
}
```

### Module Initialization

```javascript
const documentProcessor = new DocumentProcessor();
```
- Creates instance of DocumentProcessor class
- Sets up text splitters (markdown and recursive)
- Ready to process PDFs and text files

```javascript
const vectorStore = new VectorStoreAndEmbedding();
```
- Creates vector database connection
- Initializes Ollama embedding model
- Connects to ChromaDB server

**What happens during initialization**:
```javascript
constructor() {
    // 1. Create embedding model connection
    this.embeddings = new OllamaEmbeddings({
        model: 'nomic-embed-text',
        baseUrl: 'http://localhost:11434'
    });

    // 2. Connect to ChromaDB
    this.vectorstore = await Chroma.fromExistingCollection(
        this.embeddings,
        { collectionName: 'my_documents' }
    );
}
```

```javascript
const fileManager = new FileManager(documentProcessor, vectorStore);
```
- **Dependency Injection pattern**
- FileManager receives its dependencies instead of creating them
- Makes testing easier (can inject mocks)
- Promotes loose coupling

```javascript
const llmManager = new LLMManager('llama3.2', 0.7);
```
- Creates LLM interface
- Model: llama3.2 (Ollama model name)
- Temperature: 0.7 (controls randomness)
  - 0.0 = deterministic, same answer every time
  - 1.0 = very creative, different answers
  - 0.7 = balanced

```javascript
const chatManager = new ChatSession(llmManager, vectorStore);
```
- Creates session manager
- Will store active chat sessions in memory
- Uses Map for O(1) lookup by session ID

### Route Handler Example

Let's examine one route in detail:

```javascript
app.post('/upload/file', upload.single('file'), async (req, res) => {
```
**Breaking it down**:
- `app.post()` - Register a POST route
- `'/upload/file'` - The URL path
- `upload.single('file')` - Middleware: expects one file with field name 'file'
- `async (req, res) => {}` - Asynchronous route handler function

**Request object (`req`)**:
```javascript
{
    file: {
        buffer: <Buffer>,
        originalname: 'document.pdf',
        mimetype: 'application/pdf',
        size: 52480
    },
    body: {},  // Any form fields
    params: {},  // URL parameters
    query: {}  // Query string
}
```

**Response object (`res`)**:
```javascript
{
    json(data): // Send JSON response
    status(code): // Set HTTP status code
    send(data): // Send any response
    sendFile(path): // Send file
}
```

```javascript
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file provided' });
        }
```
- Validation: Check if file was uploaded
- If not, return 400 Bad Request status
- `return` prevents further execution

```javascript
        const result = await fileManager.uploadFile(req.file);
```
- Call FileManager to process the file
- `await` pauses until processing completes
- Returns object like: `{ success: true, chunks_created: 15 }`

```javascript
        const statusCode = result.success ? 200 : 400;
        res.status(statusCode).json(result);
```
- Ternary operator: `condition ? ifTrue : ifFalse`
- If successful: 200 OK
- If failed: 400 Bad Request
- Send result as JSON

```javascript
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});
```
- Catch any unexpected errors
- 500 Internal Server Error
- Return error message to client

---

## Vector Store Module Explained

### Constructor

```javascript
export class VectorStoreAndEmbedding {
    constructor() {
        this.embeddings = new OllamaEmbeddings({
            model: 'nomic-embed-text',
            baseUrl: 'http://localhost:11434',
        });
```

**What is OllamaEmbeddings?**
- LangChain wrapper for Ollama's embedding API
- Converts text to vectors
- HTTP client that calls Ollama server

**What does 'nomic-embed-text' do?**
```
Input text: "How to authenticate users"
        ↓
Ollama embedding model
        ↓
Output: [0.234, -0.567, 0.891, ..., 0.123]  (768 numbers)
```

**Why 768 dimensions?**
- Balance between accuracy and performance
- More dimensions = more precise but slower
- 768 is standard for many models

```javascript
        this.vectorstore = null;
        this.initializeVectorStore();
    }
```
- Set vectorstore to null initially
- Call async initialization function
- Can't use await in constructor, so separate method

### Initialize Vector Store

```javascript
    async initializeVectorStore() {
        try {
            this.vectorstore = await Chroma.fromExistingCollection(
                this.embeddings,
                {
                    collectionName: 'my_documents',
                    url: 'http://localhost:8000',
                }
            );
```

**What this does**:
1. Connect to ChromaDB server at localhost:8000
2. Try to load existing collection named 'my_documents'
3. If found, use it
4. If not found, throw error and create new one

**ChromaDB HTTP API calls**:
```javascript
// Behind the scenes:
GET http://localhost:8000/api/v1/collections/my_documents
// If exists: Returns collection metadata
// If not: Returns 404
```

```javascript
            console.log('✅ Vector store initialized');
        } catch (error) {
            console.log('Creating new collection...');
            this.vectorstore = await Chroma.fromDocuments(
                [],  // Empty array (no initial documents)
                this.embeddings,
                {
                    collectionName: 'my_documents',
                    url: 'http://localhost:8000',
                }
            );
```

**Fallback behavior**:
- If collection doesn't exist, create it
- Start with empty collection
- Ready to accept new documents

### Store Chunks Method

```javascript
    async storeChunks(chunks) {
        if (!chunks || chunks.length === 0) {
            return { error: 'No chunks provided' };
        }
```
- Guard clause: Early return if no chunks
- Prevents unnecessary processing

```javascript
        const documents = [];
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            if (typeof chunk === 'string') {
                const doc = new Document({
                    pageContent: chunk,
                    metadata: { chunk_index: i },
                });
                documents.push(doc);
```

**Type normalization**:
- Accept both strings and Document objects
- Convert strings to proper Document format
- Ensures consistent data structure

**Document structure**:
```javascript
{
    pageContent: "Actual text content here...",
    metadata: {
        chunk_index: 0,
        source: "file.pdf",
        type: "document"
    }
}
```

```javascript
            } else if (chunk instanceof Document ||
                      (chunk.pageContent && chunk.metadata)) {
                documents.push(chunk);
            } else {
                console.log(`Warning: Skipping invalid chunk type`);
            }
        }
```
- `instanceof` checks if object is Document class
- Fallback: Duck typing (if it has pageContent and metadata, treat as Document)
- Skip invalid chunks with warning

```javascript
        try {
            if (!this.vectorstore) {
                await this.initializeVectorStore();
            }
```
- Safety check: Ensure vectorstore is initialized
- Lazy initialization if needed

```javascript
            await this.vectorstore.addDocuments(documents);
```

**What happens here** (step by step):

1. **Generate embeddings** for each document:
```javascript
for (const doc of documents) {
    const embedding = await this.embeddings.embedQuery(doc.pageContent);
    // HTTP POST to Ollama:
    // {
    //   "model": "nomic-embed-text",
    //   "prompt": "Actual text content here..."
    // }
    // Response: [0.234, -0.567, ..., 0.123]
}
```

2. **Store in ChromaDB**:
```javascript
// HTTP POST to ChromaDB:
// {
//   "ids": ["uuid-1", "uuid-2", ...],
//   "embeddings": [[0.234, ...], [0.567, ...], ...],
//   "metadatas": [{source: "file.pdf"}, ...],
//   "documents": ["text 1", "text 2", ...]
// }
```

3. **ChromaDB indexes**:
```
- Stores embeddings in HNSW index for fast search
- Stores metadata for filtering
- Stores original text for retrieval
```

```javascript
            console.log(`Stored ${documents.length} chunks in vector database`);
            return {
                data: 'Chunks embedded and stored successfully',
                count: documents.length,
            };
        } catch (error) {
            console.error('Error storing chunks:', error);
            throw error;
        }
    }
```
- Log success
- Return confirmation
- Propagate errors up the call stack

### Search Method

```javascript
    async search(query, k = 5) {
        try {
            if (!this.vectorstore) {
                await this.initializeVectorStore();
            }

            const results = await this.vectorstore.maxMarginalRelevanceSearch(
                query,
                { k }
            );
            return results;
```

**maxMarginalRelevanceSearch explained**:

**Step 1**: Embed query
```javascript
queryEmbedding = await embeddings.embedQuery("How to authenticate users");
// [0.145, -0.523, 0.876, ...]
```

**Step 2**: Find similar vectors
```javascript
// Calculate cosine similarity with all vectors in database
similarities = [];
for (each vector in database) {
    similarity = cosineSimilarity(queryEmbedding, vector);
    similarities.push({ document, similarity });
}
// Sort by similarity descending
similarities.sort((a, b) => b.similarity - a.similarity);
```

**Step 3**: Apply MMR (Maximum Marginal Relevance)
```
Goal: Balance relevance with diversity

Algorithm:
1. Pick most relevant document (highest similarity)
2. For remaining documents:
   - Score = λ * similarity - (1-λ) * max_similarity_to_selected
   - Pick document with highest score
3. Repeat until k documents selected

Why? Prevents returning 5 nearly identical chunks
```

**Example**:
```javascript
// Without MMR (might get redundant results):
Results: [
    "JWT tokens provide authentication...",
    "JWT tokens are used for authentication...",
    "Authentication with JWT tokens...",
    "JWT authentication tokens...",
    "Using JWT for authentication..."
]

// With MMR (diverse results):
Results: [
    "JWT tokens provide authentication...",
    "Password hashing with bcrypt...",
    "Session management strategies...",
    "OAuth 2.0 integration...",
    "Two-factor authentication..."
]
```

---

## LLM Manager Module Explained

### Constructor and Initialization

```javascript
export class LLMManager {
    constructor(modelName = 'llama3.2', temperature = 0.7) {
        this.modelName = modelName;
        this.temperature = temperature;
        this.llm = this._initializeLLM();
    }
```
- Default parameters: If not provided, use llama3.2 and 0.7
- `_initializeLLM()` - Underscore prefix indicates private method convention

```javascript
    _initializeLLM() {
        return new Ollama({
            model: this.modelName,
            temperature: this.temperature,
            baseUrl: 'http://localhost:11434',
        });
    }
```

**Ollama class explained**:
- HTTP client for Ollama API
- When you call `llm.call(prompt)`, it sends:
```http
POST http://localhost:11434/api/generate
{
    "model": "llama3.2",
    "prompt": "Your prompt here...",
    "temperature": 0.7,
    "stream": false
}
```

### Create Conversational Chain

```javascript
    async createConversationalChain(retriever, k = 5) {
        const memory = new BufferMemory({
            memoryKey: 'chat_history',
            returnMessages: true,
            outputKey: 'answer',
        });
```

**BufferMemory explained**:
```javascript
// BufferMemory stores ALL messages in an array
{
    chatHistory: {
        messages: [
            HumanMessage("What is JWT?"),
            AIMessage("JWT stands for JSON Web Token..."),
            HumanMessage("How does it work?"),
            AIMessage("It works by...")
        ]
    }
}

// When creating prompt, it formats as:
"Chat History:
Human: What is JWT?
AI: JWT stands for JSON Web Token...
Human: How does it work?

Current Question: [new question]"
```

**Parameters**:
- `memoryKey`: Variable name in prompt template
- `returnMessages`: Return as Message objects (not strings)
- `outputKey`: Which field contains the answer

```javascript
        const chain = ConversationalRetrievalQAChain.fromLLM(
            this.llm,
            retriever,
            {
                memory,
                returnSourceDocuments: true,
                verbose: false,
            }
        );
```

**ConversationalRetrievalQAChain explained**:

This is a pre-built LangChain component that:

1. **Takes user question**
2. **Uses memory to maintain context**
3. **Retrieves relevant documents** using retriever
4. **Combines** question + history + documents
5. **Sends to LLM**
6. **Returns answer + sources**

**Internal flow**:
```javascript
async call({ question }) {
    // 1. Get chat history from memory
    const history = memory.loadMemoryVariables();

    // 2. Retrieve relevant documents
    const docs = await retriever.getRelevantDocuments(question);

    // 3. Format prompt
    const prompt = `
    Context from documents:
    ${docs.map(d => d.pageContent).join('\n\n')}

    Chat History:
    ${history.chat_history}

    Question: ${question}

    Answer:`;

    // 4. Call LLM
    const answer = await llm.call(prompt);

    // 5. Save to memory
    memory.saveContext({ input: question }, { output: answer });

    // 6. Return result
    return { answer, sourceDocuments: docs };
}
```

### Review Code Direct

```javascript
    async reviewCodeDirect(code, language, source, reviewType = 'quick') {
        try {
            const promptTemplate = getReviewPromptTemplate(reviewType);
```
- Get appropriate prompt template (quick, comprehensive, etc.)
- Returns LangChain PromptTemplate object

```javascript
            const prompt = await promptTemplate.format({
                code,
                language,
                source,
                context: 'No additional context available (direct review)',
                chat_history: '',
                question: 'Please review this code.',
                start_line: '',
                end_line: '',
            });
```

**format() method explained**:

Takes template with placeholders:
```
"Review this {language} code from {source}:
{code}"
```

Replaces with actual values:
```
"Review this javascript code from login.js:
function login(user, pass) { ... }"
```

**Why empty strings for some fields?**
- Template expects all variables
- Not all are relevant for direct review
- Empty strings satisfy template requirements

```javascript
            const response = await this.llm.call(prompt);
            return response;
```
- Send complete prompt to Ollama
- Wait for generated review
- Return as string

---

## Chat Session Module Explained

### Session Storage

```javascript
export class ChatSession {
    constructor(llmManager, vectorStore) {
        this.llmManager = llmManager;
        this.vectorStore = vectorStore;
        this.sessions = new Map();
    }
```

**Why Map instead of Object?**

```javascript
// Map advantages:
sessions.set(sessionId, data);  // O(1) insertion
sessions.get(sessionId);  // O(1) lookup
sessions.has(sessionId);  // O(1) check
sessions.delete(sessionId);  // O(1) deletion
sessions.size;  // O(1) count

// Can iterate:
for (const [id, session] of sessions.entries()) {
    // Process each session
}

// Keys can be any type (not just strings)
sessions.set(userObject, data);  // Valid!
```

### Create Session

```javascript
    async createSession(sessionId = null, k = 5) {
        if (!sessionId) {
            sessionId = uuidv4();
        }
```
- If no ID provided, generate random UUID
- UUID ensures uniqueness across distributed systems

```javascript
        if (this.sessions.has(sessionId)) {
            return sessionId;
        }
```
- Idempotent: Safe to call multiple times
- If session exists, just return ID

```javascript
        const retriever = this.vectorStore.vectorstore.asRetriever({
            k,
        });
```

**asRetriever() explained**:

Converts vector store to LangChain Retriever interface:
```javascript
{
    getRelevantDocuments(query) {
        // 1. Embed query
        // 2. Search vector database
        // 3. Return top k results
    }
}
```

**Why wrap in Retriever?**
- Standardized interface for LangChain
- Can swap vector stores without changing code
- Supports filtering, metadata, etc.

```javascript
        const { chain, memory } = await this.llmManager.createConversationalChain(
            retriever,
            k
        );
```
- Creates conversational chain with retriever
- Returns both chain and memory reference
- Memory shared between chain and session

```javascript
        this.sessions.set(sessionId, {
            chain,
            memory,
            created_at: new Date(),
            last_activity: new Date(),
            message_count: 0,
        });
```

**Session object structure**:
```javascript
{
    chain: ConversationalRetrievalQAChain,  // For processing questions
    memory: BufferMemory,  // Conversation history
    created_at: Date(2025-10-21T10:00:00Z),
    last_activity: Date(2025-10-21T10:00:00Z),
    message_count: 0
}
```

### Query Method

```javascript
    async query(question, sessionId, k = 5) {
        try {
            if (!this.sessions.has(sessionId)) {
                await this.createSession(sessionId, k);
            }
```
- Auto-create session if doesn't exist
- Convenient: Client doesn't need to call /chat/new first

```javascript
            const session = this.sessions.get(sessionId);

            const result = await session.chain.call({ question });
```
- Get session from Map
- Call conversational chain with question
- Chain handles: retrieval + memory + LLM

```javascript
            session.last_activity = new Date();
            session.message_count += 1;
```
- Update metadata
- `last_activity` used for cleanup
- `message_count` for statistics

```javascript
            const sources = this.llmManager.formatSources(
                result.sourceDocuments || []
            );
```
- Format retrieved documents for response
- Truncate content, add metadata
- Make response user-friendly

```javascript
            return {
                success: true,
                answer: result.answer.trim(),
                sources,
                num_sources: (result.sourceDocuments || []).length,
                session_id: sessionId,
                message_count: session.message_count,
            };
```
- Structured response
- Includes answer, sources, and metadata
- Client can display sources to user

---

## Code Parser Module Explained

### Language Detection

```javascript
    static LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        // ... more languages
    };
```
- Static property: Shared by all instances
- Map of file extensions to language names

```javascript
    detectLanguage(filename) {
        const extension = filename.includes('.')
            ? '.' + filename.split('.').pop()
            : '';
        return CodeParser.LANGUAGE_EXTENSIONS[extension.toLowerCase()] || 'unknown';
    }
```

**How it works**:
```javascript
// Example: "MyComponent.tsx"
filename.includes('.')  // true
filename.split('.')  // ["MyComponent", "tsx"]
.pop()  // "tsx"
'.' + "tsx"  // ".tsx"
.toLowerCase()  // ".tsx"
LANGUAGE_EXTENSIONS[".tsx"]  // "typescript"
```

**Edge cases**:
```javascript
// "README" (no extension)
filename.includes('.')  // false
extension = ''
return 'unknown'

// "archive.tar.gz" (multiple dots)
filename.split('.')  // ["archive", "tar", "gz"]
.pop()  // "gz"
// Not in map, returns 'unknown'
```

### Extract Python Functions

```javascript
    extractPythonFunctions(code) {
        const structures = [];
        const lines = code.split('\n');
        let currentIndent = 0;
        let currentStructure = null;
        let startLine = 0;
```

**State machine approach**:
- Track current indentation level
- Track current structure being parsed
- Know where structure started

```javascript
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            const match = line.match(/^(\s*)(def|class)\s+(\w+)/);
```

**Regex breakdown**:
```
/^(\s*)(def|class)\s+(\w+)/

^           Start of line
(\s*)       Capture group 1: Any whitespace (indentation)
(def|class) Capture group 2: Keyword "def" or "class"
\s+         One or more spaces
(\w+)       Capture group 3: Function/class name (letters, digits, underscore)

Examples that match:
"def login(user):"
"    def validate_input(data):"
"class UserManager:"
"        class NestedClass:"

Examples that DON'T match:
"# def commented_out():"  (starts with #)
"    return def"  (def not at start of statement)
```

```javascript
            if (match) {
                if (currentStructure) {
                    structures.push({
                        type: currentStructure.type,
                        name: currentStructure.name,
                        start_line: startLine,
                        end_line: i - 1,
                        code: lines.slice(startLine, i).join('\n'),
                    });
                }
```

**When new structure found**:
1. Save previous structure (if exists)
2. `lines.slice(startLine, i)` gets lines from start to current
3. `join('\n')` reassembles into string

```javascript
                currentIndent = match[1].length;
                currentStructure = {
                    type: match[2],  // 'def' or 'class'
                    name: match[3],  // function/class name
                };
                startLine = i;
```
- Save indentation level
- Start tracking new structure
- Record starting line

```javascript
            } else if (currentStructure && line.trim() &&
                       !line.startsWith(' '.repeat(currentIndent + 1)) &&
                       !line.trim().startsWith('#')) {
                if (line[0] !== ' ' && line[0] !== '\t') {
                    // End of structure detected (dedent)
                    structures.push({...});
                    currentStructure = null;
                }
            }
        }
```

**End detection logic**:
- If processing a structure AND
- Line has content AND
- Line is not indented more than structure start AND
- Line is not a comment AND
- Line starts with no indentation
- THEN structure has ended

**Example**:
```python
def function1():  # Start at indent 0
    line1         # Indent 4
    line2         # Indent 4
    if x:         # Indent 4
        line3     # Indent 8
def function2():  # Indent 0 - DEDENT DETECTED! function1 ends
```

### Chunk Code Method

```javascript
    chunkCode(code, filename) {
        const language = this.detectLanguage(filename);
        const chunks = [];

        const imports = this.extractImports(code, language);
        const importsText = imports.join('\n');
```
- Detect language first
- Extract import statements
- Will prepend imports to chunks for context

```javascript
        if (language === 'python') {
            const structures = this.extractPythonFunctions(code);

            if (structures.length > 0) {
                for (const struct of structures) {
                    let chunkContent = struct.code;

                    if (importsText &&
                        chunkContent.length + importsText.length < this.maxChunkSize) {
                        chunkContent = importsText + '\n\n' + chunkContent;
                    }
```

**Why include imports?**:
- LLM needs context to understand code
- Imports show dependencies
- Example:
```python
# Without imports:
def hash_password(password):
    return bcrypt.hash(password)
# LLM doesn't know what bcrypt is

# With imports:
import bcrypt

def hash_password(password):
    return bcrypt.hash(password)
# LLM knows it's using bcrypt library
```

```javascript
                    chunks.push(new Document({
                        pageContent: chunkContent,
                        metadata: {
                            source: filename,
                            language,
                            content_type: 'code',
                            structure_type: struct.type,  // 'def' or 'class'
                            structure_name: struct.name,  // Function name
                            start_line: struct.start_line,
                            end_line: struct.end_line,
                        },
                    }));
                }
            }
        }
```

**Rich metadata**:
- Allows filtering: "Show only functions from auth.py"
- Provides context in search results
- Enables line-level references in code reviews

### Calculate Complexity

```javascript
    calculateComplexity(code, language) {
        const metrics = {
            lines_of_code: code.split('\n').length,
            num_functions: 0,
            num_classes: 0,
            num_imports: 0,
            cyclomatic_complexity: 0,
        };
```

**Cyclomatic complexity**:
```python
def example(x, y):  # Complexity = 1 (base)
    if x > 0:       # +1
        if y > 0:   # +1
            return x + y
        else:       # No +1 (else doesn't add complexity)
            return x
    elif x < 0:     # +1
        return -x
    else:
        return 0
# Total complexity = 4

# High complexity (>10) = hard to test
# Low complexity (<5) = easy to understand
```

```javascript
        if (language === 'python') {
            metrics.num_functions = (code.match(/^\s*def\s+\w+/gm) || []).length;
```

**Regex with global multiline**:
```javascript
/^\s*def\s+\w+/gm

g = global (find all matches, not just first)
m = multiline (^ matches start of any line, not just string)

Without gm:
"def a():\ndef b():".match(/^\s*def\s+\w+/) → ["def a()"]

With gm:
"def a():\ndef b():".match(/^\s*def\s+\w+/gm) → ["def a()", "def b()"]

|| [] = if no matches, return empty array instead of null
.length = count matches
```

---

This detailed code explanation covers the core algorithms and design patterns used throughout the system!
