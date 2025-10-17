# Technical Documentation - Dual-Mode RAG Chatbot System

**Version:** 2.0
**Last Updated:** 2024
**Purpose:** Comprehensive technical documentation explaining all code components

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Core Modules Explained](#2-core-modules-explained)
3. [Data Flow & Processing Pipeline](#3-data-flow--processing-pipeline)
4. [Design Patterns & Principles](#4-design-patterns--principles)
5. [Configuration & Setup](#5-configuration--setup)
6. [Extension Guide](#6-extension-guide)

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Flask REST API                        │
│                      (main.py)                           │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼────────┐         ┌────────▼────────┐
│  Document Mode │         │  Code Review    │
│                │         │      Mode       │
└───────┬────────┘         └────────┬────────┘
        │                           │
        └─────────────┬─────────────┘
                      │
        ┌─────────────┴─────────────┐
        │    Shared Components      │
        ├───────────────────────────┤
        │ • Vector Store (ChromaDB) │
        │ • LLM Manager (Ollama)    │
        │ • Chat Session Manager    │
        │ • Embeddings              │
        └───────────────────────────┘
```

### 1.2 Technology Stack

**Backend Framework:**
- Flask 3.0.0 - Web framework
- Flask-CORS - Cross-origin resource sharing

**AI/LLM Stack:**
- LangChain 0.1.0 - LLM orchestration framework
- Ollama - Local LLM runtime (llama3.2)
- nomic-embed-text - Embedding model

**Vector Database:**
- ChromaDB 0.4.22 - Vector storage
- HNSW - Approximate nearest neighbor search

**Document Processing:**
- PyPDF2 - PDF text extraction
- BeautifulSoup4 - HTML parsing
- yt-dlp - YouTube downloads
- Whisper - Audio transcription

### 1.3 Project Structure

```
llm-chatbot/
├── main.py                      # Flask app & API endpoints (500+ lines)
├── requirements.txt             # Python dependencies
├── API_DOCUMENTATION.md         # User-facing API docs
├── TECHNICAL_DOCUMENTATION.md   # This file
│
├── src/
│   └── modules/
│       ├── __init__.py
│       ├── file_manager.py              # File upload orchestration (260+ lines)
│       ├── document_processor.py        # Document text processing (75 lines)
│       ├── vector_store_and_embedding.py # Vector DB operations (54 lines)
│       ├── llm_manager.py               # LLM chains & prompts (380+ lines)
│       ├── chat_session.py              # Session management (260 lines)
│       ├── code_parser.py               # Code parsing & chunking (405 lines)
│       └── code_review_prompts.py       # Review prompt templates (402 lines)
│
├── test/
│   └── test_system.py           # System integration tests
│
├── uploads/                     # User uploaded files
│   └── media/                   # YouTube audio files
│
└── chroma_db/                   # Persistent vector database
```

---

## 2. Core Modules Explained

### 2.1 main.py - Flask Application Entry Point

**Purpose:** Defines REST API endpoints and initializes all modules

**Key Components:**

#### Module Initialization (Lines 14-18)
```python
document_processor = document_processor_module.DocumentProcessor()
vector_store = vector_store_module.VectorStoreAndEmbedding()
file_manager = file_manager_module.FileManager(document_processor, vector_store)
llm_manager = llm_manager_module.LLMManager(model_name="llama3.2", temperature=0.7)
chat_manager = chat_session_module.ChatSession(llm_manager, vector_store)
```

**Explanation:**
- Creates singleton instances of all core modules
- Uses dependency injection pattern (modules receive their dependencies)
- `document_processor`: Handles text extraction from documents
- `vector_store`: Manages ChromaDB vector database and embeddings
- `file_manager`: Orchestrates file upload pipeline (receives both processors)
- `llm_manager`: Manages Ollama LLM with specific model and temperature
- `chat_manager`: Handles conversation sessions (receives LLM and vector store)

#### Endpoint Categories

**Document Upload Endpoints:**

```python
@app.route('/upload/file', methods=['POST'])
def upload_file():
    result = file_manager.uploadFile()
    status_code = 200 if result.get('success') else 400
    return jsonify(result), status_code
```

**Explanation:**
- Delegates to `FileManager.uploadFile()` method
- FileManager handles: file validation → content extraction → chunking → embedding → storage
- Returns JSON with success status and metadata
- HTTP 200 for success, 400 for errors

**Code Review Endpoints:**

```python
@app.route('/review/quick', methods=['POST'])
def quick_review():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    code_content = file.read().decode('utf-8')

    parser = CodeParser()
    language = parser.detect_language(filename)

    review = llm_manager.review_code_direct(
        code=code_content,
        language=language,
        source=filename,
        review_type="quick"
    )

    return jsonify({
        'success': True,
        'review': review
    })
```

**Explanation:**
- Reads file directly without storing in vector DB (faster)
- Uses `CodeParser` to detect programming language
- Calls `LLMManager.review_code_direct()` for analysis
- Returns review feedback as JSON

**Chat Endpoint:**

```python
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data['question']
    session_id = data.get('session_id', str(uuid.uuid4()))
    k = data.get('k', 5)

    result = chat_manager.query(question, session_id, k=k)
    return jsonify(result)
```

**Explanation:**
- Accepts JSON with question, optional session_id, and k (retrieval count)
- Auto-generates session_id if not provided using UUID4
- Delegates to `ChatSession.query()` which:
  1. Retrieves k relevant chunks from vector DB
  2. Passes to LLM with conversation history
  3. Returns answer with sources

---

### 2.2 file_manager.py - File Upload Orchestration

**Purpose:** Manages file uploads and orchestrates processing pipeline

**Class Structure:**

```python
class FileManager:
    def __init__(self, document_processor, vector_store, code_parser=None):
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.code_parser = code_parser if code_parser else CodeParser()
```

**Explanation:**
- Receives dependencies via constructor (Dependency Injection pattern)
- Creates CodeParser if not provided (lazy initialization)
- Maintains references to processors for delegation

#### Method: uploadFile() (Lines 25-95)

**Flow Diagram:**
```
File Upload
    ↓
Check if Code File?
    ├─ Yes → CodeParser.chunk_code()
    │         ↓
    │    Store in Vector DB
    │         ↓
    │    Return {file_type: 'code', language, complexity}
    │
    └─ No → DocumentProcessor.extract_text()
              ↓
         DocumentProcessor.chunk_document()
              ↓
         Store in Vector DB
              ↓
         Return {file_type: 'document', chunks}
```

**Code Explanation:**

```python
# Get uploaded file from Flask request
uploaded_file = request.files['file']
filename = uploaded_file.filename
destination = 'uploads/' + uploaded_file.filename
uploaded_file.save(destination)

# Check if it's a code file
if self.code_parser.is_code_file(filename):
    # Read code content
    with open(destination, 'r', encoding='utf-8') as f:
        code_content = f.read()

    # Use code-specific chunking
    chunks = self.code_parser.chunk_code(code_content, filename)

    # Store in vector database
    store_result = self.vector_store.store_chunks(chunks)

    return {
        'success': True,
        'file_type': 'code',
        'language': language,
        'chunks_created': store_result['count']
    }
```

**Key Points:**
- Saves file to `uploads/` directory first
- Uses `CodeParser.is_code_file()` to detect file type by extension
- Code files: Read as UTF-8, use syntax-aware chunking
- Document files: Extract text based on extension (.pdf, .txt, .md)
- All chunks stored in same vector database (enables cross-searching)

#### Method: uploadCodeForReview() (Lines 97-141)

**Purpose:** Upload code with complexity analysis

```python
def uploadCodeForReview(self):
    # Verify it's a code file
    if not self.code_parser.is_code_file(filename):
        return {'error': 'File is not a supported code file'}

    # Calculate complexity metrics
    complexity = self.code_parser.calculate_complexity(code_content, language)

    # Chunk and store
    chunks = self.code_parser.chunk_code(code_content, filename)
    store_result = self.vector_store.store_chunks(chunks)

    return {
        'complexity': complexity  # Includes LOC, functions, classes, etc.
    }
```

**Explanation:**
- Validates file is a code file (rejects documents)
- Calculates metrics: lines of code, function count, cyclomatic complexity
- Returns detailed complexity analysis for display to user

#### Method: uploadMediaFile() (Lines 102-154)

**Purpose:** YouTube video transcription pipeline

**Flow:**
```
YouTube URL
    ↓
downloadYouTubeFile()
    ├─ yt-dlp: Download best audio
    └─ FFmpeg: Convert to MP3
    ↓
transcribeAudioFile()
    ├─ Whisper: Transcribe to text
    └─ Create Document with metadata
    ↓
Save transcription to .txt file
    ↓
Chunk text
    ↓
Store in vector DB
```

**Code Explanation:**

```python
# Download audio
audio_file_path = self.downloadYouTubeFile(save_dir, url)

# Transcribe using Whisper
whisper_model = whisper.load_model("base")
result = whisper_model.transcribe(audio_file_path)

# Create document with metadata
doc = Document(
    page_content=result["text"],
    metadata={"source": url, "file": audio_filename}
)

# Save transcription to file for reference
with open(transcription_path, 'w', encoding='utf-8') as f:
    f.write(doc.page_content)

# Chunk and store
chunks = self.document_processor.chunkDocument(content=doc.page_content)
self.vector_store.store_chunks(chunks)
```

**Key Points:**
- Uses `yt-dlp` library for robust YouTube downloads (handles various formats)
- Extracts best audio quality available
- Uses OpenAI Whisper "base" model (balance of speed/accuracy)
- Saves transcription locally for user reference
- Processes transcription through standard chunking pipeline

#### Method: webFileUpload() (Lines 156-215)

**Purpose:** Web scraping and content extraction

```python
# Load web content using LangChain loader
loader = WebBaseLoader(url)
data = loader.load()
doc = data[0]

# Create safe filename from page title
title = doc.metadata.get('title', 'web_content')
safe_filename = title.replace(" ", "_").replace("/", "_")
safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in ('_', '-', '.'))

# Save page content with metadata
with open(destination, 'w', encoding='utf-8') as f:
    f.write(f"Source: {doc.metadata.get('source')}\n")
    f.write(f"Title: {doc.metadata.get('title')}\n")
    f.write(doc.page_content)

# Chunk and store
chunks = self.document_processor.chunkDocument(content=doc.page_content)
self.vector_store.store_chunks(chunks)
```

**Explanation:**
- LangChain's `WebBaseLoader` handles HTML parsing automatically
- Extracts metadata: title, description, language
- Sanitizes filename to prevent directory traversal attacks
- Preserves metadata in saved file for context
- Full content available for retrieval and display

---

### 2.3 code_parser.py - Code Analysis & Chunking

**Purpose:** Language-aware code parsing and intelligent chunking

**Class Structure:**

```python
class CodeParser:
    # Supported languages mapping
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        # ... 20+ languages
    }

    def __init__(self, max_chunk_size=2000):
        self.max_chunk_size = max_chunk_size
```

#### Method: detect_language() (Lines 47-55)

```python
def detect_language(self, filename: str) -> str:
    extension = '.' + filename.split('.')[-1] if '.' in filename else ''
    return self.LANGUAGE_EXTENSIONS.get(extension.lower(), 'unknown')
```

**Explanation:**
- Extracts file extension from filename
- Looks up in LANGUAGE_EXTENSIONS dictionary
- Returns 'unknown' for unsupported types (used for validation)
- Case-insensitive matching

#### Method: extract_python_functions() (Lines 65-124)

**Purpose:** Parse Python code to identify functions and classes

**Algorithm:**
```
Initialize: structures = [], current_indent = 0

For each line:
    If line matches "def funcname" or "class ClassName":
        ├─ Save previous structure (if exists)
        ├─ Record indent level
        ├─ Start new structure
        │
    Else if dedent detected (back to indent 0):
        └─ End current structure

Return: List of {type, name, start_line, end_line, code}
```

**Code Explanation:**

```python
for i, line in enumerate(lines):
    # Detect function or class definition using regex
    match = re.match(r'^(\s*)(def|class)\s+(\w+)', line)

    if match:
        # Save previous structure if exists
        if current_structure:
            structures.append({
                'type': current_structure['type'],
                'name': current_structure['name'],
                'start_line': start_line,
                'end_line': i - 1,
                'code': '\n'.join(lines[start_line:i])
            })

        # Start new structure
        current_indent = len(match.group(1))  # Count leading spaces
        current_structure = {
            'type': match.group(2),  # 'def' or 'class'
            'name': match.group(3)   # Function/class name
        }
        start_line = i
```

**Key Points:**
- Uses regex to match Python syntax: `def name` or `class Name`
- Tracks indentation level to detect structure boundaries
- Preserves complete function/class code including docstrings
- Returns structured metadata for each code unit

#### Method: chunk_code() (Lines 183-242)

**Purpose:** Intelligently chunk code while preserving structure

**Strategy Decision Tree:**
```
Is Python?
    ├─ Yes: Try extract_python_functions()
    │       ├─ Found structures? → Chunk by function/class
    │       └─ No structures? → Chunk by size
    │
    └─ No: Is JavaScript/TypeScript?
            ├─ Yes: Try extract_javascript_functions()
            │       └─ Chunk by size with structure awareness
            │
            └─ No: Generic chunking by line count
```

**Code Explanation:**

```python
def chunk_code(self, code: str, filename: str) -> List[Document]:
    language = self.detect_language(filename)
    chunks = []

    # Extract imports to include as context
    imports = self.extract_imports(code, language)
    imports_text = '\n'.join(imports)

    # Language-specific chunking
    if language == 'python':
        structures = self.extract_python_functions(code)

        if structures:
            # Chunk by function/class
            for struct in structures:
                chunk_content = struct['code']

                # Prepend imports if space allows
                if len(chunk_content) + len(imports_text) < self.max_chunk_size:
                    chunk_content = imports_text + '\n\n' + chunk_content

                chunks.append(Document(
                    page_content=chunk_content,
                    metadata={
                        'source': filename,
                        'language': language,
                        'content_type': 'code',
                        'structure_type': struct['type'],  # 'def' or 'class'
                        'structure_name': struct['name'],
                        'start_line': struct['start_line'],
                        'end_line': struct['end_line']
                    }
                ))
        else:
            # Fall back to size-based chunking
            chunks = self._chunk_by_size(code, filename, language)

    return chunks
```

**Key Design Decisions:**

1. **Include Imports:** Each chunk gets necessary imports for context
2. **Preserve Boundaries:** Functions/classes never split mid-definition
3. **Rich Metadata:** Stores structure type, name, line numbers for precise feedback
4. **Fallback Strategy:** If structure detection fails, uses line-based chunking
5. **Size Limits:** Respects max_chunk_size (default 2000 chars) for LLM context window

#### Method: _chunk_by_size() (Lines 244-292)

**Purpose:** Line-aware size-based chunking (fallback strategy)

```python
def _chunk_by_size(self, code: str, filename: str, language: str):
    chunks = []
    lines = code.split('\n')
    current_chunk = []
    current_size = 0
    start_line = 0

    for i, line in enumerate(lines):
        line_size = len(line) + 1  # +1 for newline

        # Check if adding this line exceeds max size
        if current_size + line_size > self.max_chunk_size and current_chunk:
            # Save current chunk
            chunks.append(Document(
                page_content='\n'.join(current_chunk),
                metadata={
                    'source': filename,
                    'language': language,
                    'start_line': start_line,
                    'end_line': i - 1
                }
            ))

            # Start new chunk
            current_chunk = []
            current_size = 0
            start_line = i

        current_chunk.append(line)
        current_size += line_size

    return chunks
```

**Explanation:**
- Never splits in middle of a line (preserves syntax)
- Tracks character count for size limit
- Maintains line number ranges in metadata
- Ensures no empty chunks

#### Method: calculate_complexity() (Lines 294-327)

**Purpose:** Basic code quality metrics

```python
def calculate_complexity(self, code: str, language: str) -> Dict:
    metrics = {
        'lines_of_code': len(code.split('\n')),
        'num_functions': 0,
        'num_classes': 0,
        'num_imports': 0,
        'cyclomatic_complexity': 0
    }

    if language == 'python':
        metrics['num_functions'] = len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))
        metrics['num_classes'] = len(re.findall(r'^\s*class\s+\w+', code, re.MULTILINE))
        metrics['num_imports'] = len(re.findall(r'^(?:from\s+[\w.]+\s+)?import\s+', code, re.MULTILINE))

        # Simple cyclomatic complexity (count decision points)
        metrics['cyclomatic_complexity'] = len(re.findall(
            r'\b(if|elif|for|while|except|and|or)\b', code
        ))

    return metrics
```

**Metrics Explained:**

- **lines_of_code:** Total line count (including comments/blanks)
- **num_functions:** Count of function definitions
- **num_classes:** Count of class definitions
- **num_imports:** Count of import statements
- **cyclomatic_complexity:** Simplified metric counting decision points (if, for, while, etc.)

**Note:** This is a basic implementation. Production systems use tools like `radon` for accurate McCabe complexity.

---

### 2.4 document_processor.py - Document Text Processing

**Purpose:** Extract and chunk text from documents

**Class Structure:**

```python
class DocumentProcessor:
    def __init__(self):
        # Two-stage text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n\n", "\n\n", "\n", ".", " ", ""]
        )

        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
        )
```

**Explanation:**

**RecursiveCharacterTextSplitter:**
- Tries to split on hierarchical separators (paragraph → sentence → word → character)
- `chunk_size=1000`: Target size for each chunk
- `chunk_overlap=100`: 10% overlap preserves context between chunks
- Hierarchical separators maintain document structure

**MarkdownHeaderTextSplitter:**
- Markdown-aware splitting preserves heading hierarchy
- Keeps sections together under their headers
- Metadata includes header level and content

#### Method: pdfToText() (Lines 27-37)

```python
def pdfToText(self, file_path, filename):
    reader = PyPDF2.PdfReader(file_path)
    content = ""

    for page in reader.pages:
        content += page.extract_text() + "\n\n"

    return content
```

**Explanation:**
- Uses PyPDF2 for PDF parsing
- Extracts text from all pages sequentially
- Adds double newline between pages (preserves structure for splitting)
- Returns concatenated text

**Limitations:**
- No OCR (can't read scanned/image PDFs)
- No table extraction
- Formatting may be lost

#### Method: chunkDocument() (Lines 50-74)

**Purpose:** Two-stage chunking strategy

**Flow:**
```
Document Text
    ↓
Stage 1: Markdown Splitter
    ├─ Splits on headers (#, ##)
    └─ Preserves document hierarchy
    ↓
Stage 2: Recursive Character Splitter
    ├─ Splits large chunks further
    └─ Uses hierarchical separators
    ↓
List of Document objects with metadata
```

**Code:**

```python
def chunkDocument(self, content, metadata=None):
    # Stage 1: Markdown-aware splitting
    md_docs = self.md_splitter.split_text(content)

    # Stage 2: Recursive splitting (handles large sections)
    final_chunks = self.text_splitter.split_documents(md_docs)

    # Add custom metadata
    for chunk in final_chunks:
        if metadata:
            chunk.metadata.update(metadata)

    print(f"Created {len(final_chunks)} chunks")
    return final_chunks
```

**Why Two Stages?**
1. **Stage 1** preserves document structure (headers, sections)
2. **Stage 2** ensures no chunk exceeds size limit (for LLM context window)
3. Metadata preserved through both stages

---

### 2.5 vector_store_and_embedding.py - Vector Database

**Purpose:** Manage embeddings and vector storage

**Class Structure:**

```python
class VectorStoreAndEmbedding:
    def __init__(self):
        # Initialize embeddings model
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Initialize ChromaDB with persistence
        self.vectorstore = Chroma(
            collection_name="my_documents",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
```

**Components Explained:**

**OllamaEmbeddings:**
- Uses `nomic-embed-text` model (768-dimensional embeddings)
- Converts text to vector representations
- Enables semantic similarity search

**ChromaDB:**
- Open-source vector database
- Persistent storage (survives server restarts)
- Collection "my_documents" stores all chunks (code + documents)
- Automatically handles indexing via HNSW algorithm

#### Method: store_chunks() (Lines 20-48)

```python
def store_chunks(self, chunks):
    if isinstance(chunks, str):
        # Convert string to Document
        chunks = [Document(page_content=chunks)]

    elif isinstance(chunks, list) and len(chunks) > 0:
        if isinstance(chunks[0], str):
            # Convert list of strings to Documents
            chunks = [Document(page_content=chunk) for chunk in chunks]

    # Store in ChromaDB
    self.vectorstore.add_documents(chunks)

    # Get updated count
    count = self.vectorstore._collection.count()
    print(f"Stored {len(chunks)} chunks. Total in DB: {count}")

    return {'count': len(chunks), 'total': count}
```

**Explanation:**
- Accepts flexible input: string, list of strings, or Document objects
- Normalizes to Document objects (required by ChromaDB)
- `add_documents()` automatically:
  1. Generates embeddings for each chunk
  2. Stores vectors + metadata in collection
  3. Updates index for fast retrieval
- Returns count for user feedback

**Under the Hood:**
1. Each chunk's text is passed to embedding model
2. Model returns 768-dimensional vector
3. Vector stored with metadata in ChromaDB
4. HNSW index updated for O(log n) search

#### Method: search() (Lines 50-53)

```python
def search(self, query, k=5):
    return self.vectorstore.max_marginal_relevance_search(query, k=k)
```

**Search Strategy: MMR (Maximal Marginal Relevance)**

**Why MMR instead of simple similarity?**
- **Similarity Search:** Returns k most similar chunks (may be redundant)
- **MMR:** Balances similarity AND diversity
  - Finds similar chunks
  - Filters out redundant/duplicate content
  - Returns diverse set of relevant results

**Algorithm:**
```
1. Generate embedding for query
2. Find top 2k most similar chunks
3. Select first chunk (most similar)
4. For remaining chunks:
   - Score = similarity_to_query - λ * max_similarity_to_selected
   - Higher λ = more diversity
   - Select chunk with highest score
5. Repeat until k chunks selected
```

**Parameters:**
- `query`: User's question (embedded automatically)
- `k`: Number of chunks to return (default 5)

**Returns:** List of Document objects with content + metadata

---

### 2.6 llm_manager.py - LLM Chain Management

**Purpose:** Create and manage LLM chains for different use cases

**Class Structure:**

```python
class LLMManager:
    def __init__(self, model_name="llama3.2", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        return Ollama(
            model=self.model_name,
            temperature=self.temperature
        )
```

**Parameters Explained:**

**model_name:**
- Specifies which Ollama model to use
- Default: "llama3.2" (general-purpose, good for code + text)
- Can be changed to other models (qwen, mistral, codellama, etc.)

**temperature:**
- Controls randomness in responses (0.0 to 1.0)
- 0.7: Balanced creativity and consistency
- Lower (0.1-0.3): More deterministic, better for code
- Higher (0.8-1.0): More creative, better for writing

#### Method: create_conversational_chain() (Lines 30-57)

**Purpose:** Create RAG chain with conversation memory

```python
def create_conversational_chain(self, retriever, k=5):
    # Create memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # Create the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=self.llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    return chain, memory
```

**Components:**

**ConversationBufferMemory:**
- Stores entire chat history in memory
- `memory_key="chat_history"`: Variable name in prompts
- `return_messages=True`: Returns as message objects (not strings)
- `output_key='answer'`: Which field to store from responses

**ConversationalRetrievalChain:**
- Combines retrieval + LLM + memory
- **Flow:**
  ```
  User Question
      ↓
  Retriever.search(question) → Get k relevant chunks
      ↓
  Load chat_history from memory
      ↓
  Format prompt: "Context: {chunks}\nHistory: {chat_history}\nQuestion: {question}"
      ↓
  LLM generates answer
      ↓
  Store Q&A in memory
      ↓
  Return {answer, source_documents}
  ```

**Why return both chain and memory?**
- Chain is stateless (doesn't hold state)
- Memory must be stored externally (in ChatSession)
- Allows multiple sessions with separate histories

#### Method: create_code_review_chain() (Lines 156-188)

**Purpose:** Code review with specialized prompts

```python
def create_code_review_chain(self, retriever, review_type="conversational", k=5):
    # Get specialized prompt template
    prompt = get_review_prompt_template(review_type)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=self.llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},  # Custom prompt!
        verbose=False
    )

    return chain, memory
```

**Key Difference from create_conversational_chain():**
- Uses `combine_docs_chain_kwargs={"prompt": prompt}`
- Replaces default Q&A prompt with specialized code review prompt
- Prompt includes instructions for code analysis

**Available review_types:**
- "comprehensive": Full code review
- "quick": Fast critical issues
- "security": Security vulnerabilities
- "performance": Performance optimization
- "best_practices": Design patterns
- "bug_detection": Find bugs
- "explanation": Explain code
- "improvement": Suggest refactorings

#### Method: review_code_direct() (Lines 190-225)

**Purpose:** Review code without RAG (direct LLM call)

**Use Case:** Quick review of single file without codebase context

```python
def review_code_direct(self, code, language, source, review_type="quick"):
    # Get prompt template
    prompt_template = get_review_prompt_template(review_type)

    # Format prompt with code info
    prompt = prompt_template.format(
        code=code,
        language=language,
        source=source,
        context="No additional context available (direct review)",
        chat_history="",
        question="Please review this code.",
        start_line="",
        end_line=""
    )

    # Direct LLM call (no retrieval)
    response = self.llm(prompt)
    return response
```

**Explanation:**
- Bypasses vector DB retrieval (faster)
- Fills all template variables
- Empty strings for unavailable fields
- Good for: one-off reviews, small files, no related code

#### Method: review_code_with_context() (Lines 227-269)

**Purpose:** Review code with related code from vector DB

**Use Case:** Review file in context of larger codebase

```python
def review_code_with_context(self, code, language, source,
                              context_documents, question, review_type):
    # Format context from retrieved documents
    context = format_code_context(context_documents)

    # Get prompt template
    prompt_template = get_review_prompt_template(review_type)

    # Format prompt with all information
    prompt = prompt_template.format(
        code=code,
        language=language,
        source=source,
        context=context,  # Related code from vector DB
        chat_history="",
        question=question,
        start_line="",
        end_line=""
    )

    response = self.llm(prompt)
    return response
```

**format_code_context() helper:**
```python
def format_code_context(documents, max_context_length=1500):
    context_parts = []

    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        language = doc.metadata.get('language', 'unknown')
        start_line = doc.metadata.get('start_line', '')

        # Format with metadata
        part = f"\n**From {source} (lines {start_line}):**\n```{language}\n{doc.page_content[:500]}\n```\n"

        if len(context_parts) + len(part) < max_context_length:
            context_parts.append(part)

    return "\n".join(context_parts)
```

**Explanation:**
- Formats retrieved code chunks as markdown
- Includes source file and line numbers
- Limits total context (prevents token overflow)
- Truncates individual chunks at 500 chars

#### Methods: explain_code(), suggest_improvements(), detect_bugs() (Lines 271-381)

**Purpose:** Specialized code analysis functions

**All follow same pattern:**
```python
def explain_code(self, code, language, source, context_documents, question):
    # 1. Format context
    context = format_code_context(context_documents) if context_documents else "No context"

    # 2. Get specialized prompt
    prompt_template = get_review_prompt_template("explanation")

    # 3. Format prompt
    prompt = prompt_template.format(
        code=code,
        language=language,
        source=source,
        context=context,
        question=question
    )

    # 4. Call LLM
    response = self.llm(prompt)
    return response
```

**Key Differences:**
- **explain_code():** Uses "explanation" prompt template, focuses on "what does this do?"
- **suggest_improvements():** Uses "improvement" prompt, asks "how to make better?"
- **detect_bugs():** Uses "bug_detection" prompt, analyzes for errors

**Design Pattern:** Template Method Pattern
- Common structure (format context → get prompt → call LLM)
- Variation in prompt template only
- Easy to add new analysis types

---

### 2.7 code_review_prompts.py - Prompt Templates

**Purpose:** Define specialized prompts for different code review types

**Structure:**

```python
# Prompt as Python string
COMPREHENSIVE_CODE_REVIEW_PROMPT = """
You are an expert code reviewer...

Review the following code:
```{language}
{code}
```

Analyze for:
1. Code Quality
2. Security
3. Performance
...
"""

# Function to get prompt template
def get_review_prompt_template(review_type: str) -> PromptTemplate:
    prompts = {
        "comprehensive": COMPREHENSIVE_CODE_REVIEW_PROMPT,
        "quick": QUICK_CODE_REVIEW_PROMPT,
        # ...
    }

    template = prompts.get(review_type, COMPREHENSIVE_CODE_REVIEW_PROMPT)

    return PromptTemplate(
        template=template,
        input_variables=["code", "language", "source", "context",
                        "chat_history", "question"]
    )
```

**Template Variables:**
- `{code}`: The code being reviewed
- `{language}`: Programming language
- `{source}`: Filename
- `{context}`: Related code from codebase
- `{chat_history}`: Previous conversation
- `{question}`: Specific user question
- `{start_line}`, `{end_line}`: Line numbers (optional)

#### Prompt: COMPREHENSIVE_CODE_REVIEW_PROMPT (Lines 7-79)

**Structure:**
```
System Role: "You are an expert code reviewer..."
    ↓
Code Display: Shows code with language syntax
    ↓
Context: Related code from codebase
    ↓
Analysis Categories:
    1. Code Quality & Readability
    2. Best Practices & Design Patterns
    3. Potential Bugs
    4. Security Issues
    5. Performance Concerns
    6. Testing & Maintainability
    ↓
Output Format: Structured feedback with line references
```

**Key Elements:**

**System Role:**
```
"You are an expert code reviewer with years of experience in
software engineering best practices, security, and performance optimization."
```
- Sets expertise level
- Influences tone and depth of analysis

**Analysis Categories:**
```
1. **Code Quality & Readability**
   - Variable and function naming
   - Code organization
   - Comments and documentation

2. **Best Practices & Design Patterns**
   - Language-specific idioms
   - SOLID principles
   - DRY, KISS, YAGNI

3. **Potential Bugs**
   - Logic errors
   - Edge cases
   - Null checks

4. **Security Issues**
   - Input validation
   - SQL injection
   - XSS vulnerabilities

5. **Performance Concerns**
   - Algorithm efficiency
   - Memory usage
   - Database queries

6. **Testing & Maintainability**
   - Testability
   - Technical debt
```

**Output Instructions:**
```
"Provide specific, actionable feedback with:
- Line references where applicable
- Severity level (Critical/High/Medium/Low)
- Concrete suggestions for improvement
- Code examples for fixes"
```

#### Prompt: QUICK_CODE_REVIEW_PROMPT (Lines 82-99)

**Purpose:** Fast review for critical issues only

```
You are a code reviewer. Quickly analyze this code for critical issues:

**Code:**
```{language}
{code}
```

Focus on:
1. Critical bugs or security issues
2. Major performance problems
3. Obvious best practice violations

Provide concise, actionable feedback with line references.
```

**Differences from Comprehensive:**
- Shorter prompt (faster processing)
- Focuses on "critical" issues only
- "Concise" output requested
- No deep analysis categories

#### Prompt: SECURITY_REVIEW_PROMPT (Lines 102-129)

**Purpose:** Security-focused analysis

**Security Checklist:**
```
- Input validation issues
- SQL injection vulnerabilities
- XSS (Cross-Site Scripting) vulnerabilities
- Authentication/authorization flaws
- Sensitive data exposure
- Insecure dependencies
- Cryptography misuse
- OWASP Top 10 issues
```

**OWASP Top 10 Reference:**
1. Broken Access Control
2. Cryptographic Failures
3. Injection
4. Insecure Design
5. Security Misconfiguration
6. Vulnerable Components
7. Authentication Failures
8. Software and Data Integrity Failures
9. Security Logging Failures
10. Server-Side Request Forgery

#### Prompt: PERFORMANCE_REVIEW_PROMPT (Lines 132-159)

**Performance Analysis Areas:**
```
- Algorithm complexity (Big O notation)
- Memory usage and leaks
- Database query optimization
- I/O operations efficiency
- Caching opportunities
- Concurrency issues
- Resource management
```

**Example Output:**
```
Performance Analysis:

HIGH: Line 45 - O(n²) complexity
Current: Nested loop iterates over entire list twice
Recommendation: Use dictionary lookup (O(n))

MEDIUM: Line 78 - Database N+1 query problem
Current: Queries database inside loop (100 queries)
Recommendation: Use batch query with JOIN
```

#### Prompt: CONVERSATIONAL_CODE_REVIEW_PROMPT (Lines 193-220)

**Purpose:** Conversational tone for interactive sessions

**Key Differences:**
```
System Role:
"You are an experienced code reviewer having a conversation
with a developer about their code."

Tone Instructions:
"Keep your tone professional but friendly,
like a senior developer helping a colleague."
```

**Uses chat_history:**
```
**Previous conversation:**
{chat_history}

**Developer's question:**
{question}
```

- Maintains conversation context
- References previous messages
- Answers specific questions directly

---

### 2.8 chat_session.py - Session Management

**Purpose:** Manage multiple conversation sessions with separate histories

**Class Structure:**

```python
class ChatSession:
    def __init__(self, llm_manager, vector_store):
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.sessions = {}  # Dictionary storing all sessions
```

**Session Storage Structure:**
```python
self.sessions = {
    'session_id_1': {
        'chain': ConversationalRetrievalChain,
        'memory': ConversationBufferMemory,
        'created_at': datetime,
        'last_activity': datetime,
        'message_count': int
    },
    'session_id_2': { ... },
    ...
}
```

#### Method: create_session() (Lines 20-55)

**Purpose:** Initialize a new conversation session

```python
def create_session(self, session_id=None):
    # Auto-generate ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    # Check if session already exists
    if session_id in self.sessions:
        return session_id

    # Create retriever from vector store
    retriever = self.vector_store.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )

    # Create conversational chain
    chain, memory = self.llm_manager.create_conversational_chain(retriever)

    # Store session
    self.sessions[session_id] = {
        'chain': chain,
        'memory': memory,
        'created_at': datetime.now(),
        'last_activity': datetime.now(),
        'message_count': 0
    }

    return session_id
```

**Explanation:**

**UUID Generation:**
```python
str(uuid.uuid4())  # e.g., "550e8400-e29b-41d4-a716-446655440000"
```
- Universally unique identifier
- No coordination needed between systems
- Safe for distributed systems

**Retriever Configuration:**
```python
search_type="mmr"  # Maximal Marginal Relevance
search_kwargs={"k": 5}  # Return 5 documents
```

**Metadata Tracking:**
- `created_at`: When session started
- `last_activity`: Last message timestamp (for cleanup)
- `message_count`: Total messages exchanged

#### Method: query() (Lines 57-104)

**Purpose:** Process user question and return answer

```python
def query(self, question, session_id=None, k=5):
    # Create session if doesn't exist
    if not session_id or session_id not in self.sessions:
        session_id = self.create_session(session_id)

    # Get session
    session = self.sessions[session_id]
    chain = session['chain']

    # Run chain with question
    result = chain({"question": question})

    # Update session metadata
    session['last_activity'] = datetime.now()
    session['message_count'] += 1

    # Format sources
    sources = self.llm_manager.format_sources(result['source_documents'])

    # Return comprehensive response
    return {
        'success': True,
        'answer': result['answer'],
        'sources': sources,
        'num_sources': len(sources),
        'session_id': session_id,
        'message_count': session['message_count']
    }
```

**Flow Diagram:**
```
User Question
    ↓
Session exists?
    ├─ No → create_session()
    └─ Yes → Continue
    ↓
chain({"question": question})
    ├─ Retrieves k relevant chunks
    ├─ Loads chat history from memory
    ├─ Formats prompt
    ├─ Calls LLM
    └─ Stores Q&A in memory
    ↓
Update metadata (last_activity, message_count)
    ↓
Format sources (preview + metadata)
    ↓
Return JSON response
```

**Error Handling:**
```python
except Exception as e:
    return {
        'success': False,
        'error': str(e),
        'session_id': session_id
    }
```

#### Method: get_history() (Lines 131-167)

**Purpose:** Retrieve conversation history

```python
def get_history(self, session_id):
    if session_id not in self.sessions:
        return {'history': [], 'length': 0}

    memory = self.sessions[session_id]['memory']

    # Load messages from memory
    messages = memory.load_memory_variables({})
    chat_history = messages.get('chat_history', [])

    # Format as Q&A pairs
    history = []
    for i in range(0, len(chat_history), 2):
        if i + 1 < len(chat_history):
            history.append({
                'question': chat_history[i].content,
                'answer': chat_history[i + 1].content
            })

    return {
        'history': history,
        'length': len(history),
        'message_count': self.sessions[session_id]['message_count']
    }
```

**Memory Structure:**
- Messages stored as list: [HumanMessage, AIMessage, HumanMessage, AIMessage, ...]
- Even indices = user questions
- Odd indices = AI answers
- Paired for display

#### Method: cleanup_inactive_sessions() (Lines 231-260)

**Purpose:** Remove old sessions to free memory

```python
def cleanup_inactive_sessions(self, inactive_hours=24):
    now = datetime.now()
    threshold = timedelta(hours=inactive_hours)

    to_remove = []

    for session_id, session in self.sessions.items():
        # Calculate time since last activity
        time_diff = now - session['last_activity']

        if time_diff > threshold:
            to_remove.append(session_id)

    # Remove inactive sessions
    for session_id in to_remove:
        del self.sessions[session_id]

    return {
        'success': True,
        'cleaned': len(to_remove),
        'remaining': len(self.sessions)
    }
```

**Why Cleanup Needed?**
- Sessions stored in memory (RAM)
- Each session contains:
  - Chain object
  - Memory buffer (all messages)
  - LLM connections
- Long-running server accumulates sessions
- Cleanup prevents memory leaks

**Cleanup Strategy:**
- Default: Remove sessions inactive for 24+ hours
- Configurable threshold
- Returns count of removed sessions

---

## 3. Data Flow & Processing Pipeline

### 3.1 Document Upload Pipeline

**Complete Flow:**

```
User uploads PDF
    ↓
POST /upload/file
    ↓
FileManager.uploadFile()
    ├─ Save to uploads/ directory
    ├─ Detect file type (PDF/TXT/MD)
    └─ If PDF:
        ↓
DocumentProcessor.pdfToText()
    ├─ Extract text from all pages
    └─ Return concatenated text
        ↓
DocumentProcessor.chunkDocument()
    ├─ Stage 1: MarkdownHeaderTextSplitter
    │   └─ Preserve heading hierarchy
    ├─ Stage 2: RecursiveCharacterTextSplitter
    │   ├─ chunk_size=1000
    │   ├─ chunk_overlap=100
    │   └─ Split on: \n\n\n → \n\n → \n → . → space
    └─ Return List[Document] with metadata
        ↓
VectorStoreAndEmbedding.store_chunks()
    ├─ For each chunk:
    │   ├─ Generate embedding (768-dim vector)
    │   └─ Store in ChromaDB
    └─ Update HNSW index
        ↓
Return success response
{
  "success": true,
  "filename": "document.pdf",
  "chunks_created": 15
}
```

**Timing (approximate):**
- File upload: < 1s
- PDF text extraction: 1-5s
- Chunking: < 1s
- Embedding generation: 0.5s per chunk
- Vector storage: < 1s
- **Total: 5-20s for typical document**

### 3.2 Code Upload Pipeline

**Complete Flow:**

```
User uploads Python file
    ↓
POST /upload/code
    ↓
FileManager.uploadCodeForReview()
    ├─ Save to uploads/
    ├─ Read as UTF-8
    └─ Verify is code file
        ↓
CodeParser.detect_language()
    └─ Returns "python"
        ↓
CodeParser.calculate_complexity()
    ├─ Count functions: 5
    ├─ Count classes: 2
    ├─ Count imports: 8
    ├─ Lines of code: 150
    └─ Cyclomatic complexity: 23
        ↓
CodeParser.chunk_code()
    ├─ Extract imports (included in each chunk)
    ├─ Extract functions using regex
    │   ├─ def function1: lines 10-25
    │   ├─ def function2: lines 27-45
    │   └─ class MyClass: lines 50-100
    ├─ Create Document for each structure
    └─ Metadata: {language, structure_type, name, line_numbers}
        ↓
VectorStoreAndEmbedding.store_chunks()
    └─ Store with embeddings
        ↓
Return success with complexity metrics
{
  "success": true,
  "language": "python",
  "chunks_created": 7,
  "complexity": {...}
}
```

**Key Differences from Documents:**
- Preserves code structure (functions/classes)
- Includes line numbers in metadata
- Calculates complexity metrics
- Syntax-aware chunking

### 3.3 Chat Query Pipeline

**Complete Flow:**

```
User asks: "How does authentication work?"
    ↓
POST /chat
{
  "question": "How does authentication work?",
  "session_id": "abc123",
  "k": 5
}
    ↓
ChatSession.query()
    ├─ Session exists?
    │   ├─ No → create_session()
    │   └─ Yes → Continue
    └─ Get chain and memory
        ↓
Embedding Generation
    ├─ Question → nomic-embed-text
    └─ Returns 768-dim vector
        ↓
Vector Search (MMR)
    ├─ Search ChromaDB for similar vectors
    ├─ Find top candidates
    ├─ Apply MMR (diversity filter)
    └─ Return top 5 documents
        ↓
Load Conversation History
    ├─ Get previous Q&A from memory
    └─ Format as message list
        ↓
Prompt Construction
    ├─ System: "You are a helpful AI assistant..."
    ├─ Context: {5 retrieved chunks}
    ├─ History: {previous conversation}
    └─ Question: {user question}
        ↓
LLM Processing (Ollama)
    ├─ Send prompt to llama3.2
    ├─ Temperature: 0.7
    ├─ Max tokens: ~2048
    └─ Generate answer
        ↓
Memory Update
    ├─ Store question in memory
    └─ Store answer in memory
        ↓
Response Formatting
    ├─ Extract answer
    ├─ Format sources (filename, preview, metadata)
    └─ Update session metadata
        ↓
Return JSON
{
  "success": true,
  "answer": "Authentication in this system uses...",
  "sources": [{...}, {...}],
  "num_sources": 5,
  "session_id": "abc123",
  "message_count": 3
}
```

**Timing (approximate):**
- Embedding: 0.3s
- Vector search: 0.1s
- Memory load: < 0.1s
- LLM processing: 2-10s (depends on answer length)
- **Total: 3-15s**

### 3.4 Code Review Pipeline (Quick Review)

**Complete Flow:**

```
User uploads app.py for review
    ↓
POST /review/quick
    ↓
Read file content (no storage)
    ↓
CodeParser.detect_language()
    └─ Returns "python"
        ↓
Load QUICK_CODE_REVIEW_PROMPT
    └─ Template with analysis instructions
        ↓
Format Prompt
```{language}
{code}
```

Focus on:
1. Critical bugs
2. Security issues
3. Performance problems
```
    ↓
LLM Processing (Direct Call)
    ├─ No retrieval (faster)
    ├─ No conversation history
    └─ Single-shot analysis
        ↓
Return Review
{
  "success": true,
  "filename": "app.py",
  "language": "python",
  "review_type": "quick",
  "review": "Critical Issues:\n1. Line 45: SQL injection..."
}
```

**Timing:**
- File read: < 0.1s
- Language detection: < 0.1s
- Prompt formatting: < 0.1s
- LLM analysis: 5-15s
- **Total: 5-20s**

### 3.5 Code Review Pipeline (Comprehensive with Context)

**Complete Flow:**

```
User requests comprehensive review
    ↓
POST /review/comprehensive
    ↓
Read file content
    ↓
Vector Search for Context
    ├─ Embed first 500 chars of code
    ├─ Search for similar code chunks
    └─ Return top 3 related files
        ↓
Format Context
```
**From user_model.py (lines 15-45):**
```python
class User:
    def authenticate(self, password):
        ...
```

**From auth_utils.py (lines 8-20):**
```python
def hash_password(password):
    ...
```
```
    ↓
Load COMPREHENSIVE_CODE_REVIEW_PROMPT
    ↓
Format Complete Prompt
```
**Code to Review:**
{code}

**Context from codebase:**
{formatted context from 3 files}

**Analysis Categories:**
1. Code Quality
2. Security
3. Performance
4. Best Practices
5. Bugs
6. Testing
```
    ↓
LLM Processing
    └─ Analyzes with full context
        ↓
Return Detailed Review
{
  "success": true,
  "review_type": "comprehensive",
  "review": "Comprehensive Analysis:\n\n1. CODE QUALITY...",
  "context_used": 3
}
```

**Timing:**
- File read: < 0.1s
- Vector search: 0.1s
- Context formatting: < 0.1s
- LLM analysis: 15-45s (longer prompt)
- **Total: 15-50s**

---

## 4. Design Patterns & Principles

### 4.1 Dependency Injection

**Pattern Used Throughout:**

```python
# FileManager receives its dependencies
class FileManager:
    def __init__(self, document_processor, vector_store):
        self.document_processor = document_processor
        self.vector_store = vector_store

# Instantiation in main.py
document_processor = DocumentProcessor()
vector_store = VectorStoreAndEmbedding()
file_manager = FileManager(document_processor, vector_store)
```

**Benefits:**
- **Testability:** Easy to inject mocks for testing
- **Flexibility:** Can swap implementations (e.g., different vector DB)
- **Loose Coupling:** Modules don't create their own dependencies
- **Clear Dependencies:** Constructor shows what module needs

**Alternative (Anti-Pattern):**
```python
# BAD: FileManager creates dependencies internally
class FileManager:
    def __init__(self):
        self.document_processor = DocumentProcessor()  # Hard-coded!
        self.vector_store = VectorStoreAndEmbedding()  # Can't swap!
```

### 4.2 Strategy Pattern

**Used in:** Code chunking strategies

```python
class CodeParser:
    def chunk_code(self, code, filename):
        language = self.detect_language(filename)

        # Select strategy based on language
        if language == 'python':
            return self._chunk_python(code)
        elif language == 'javascript':
            return self._chunk_javascript(code)
        else:
            return self._chunk_generic(code)
```

**Benefits:**
- Different algorithms for different languages
- Easy to add new languages
- Encapsulates variation

### 4.3 Template Method Pattern

**Used in:** LLM review methods

```python
class LLMManager:
    # Template method
    def _review_with_template(self, code, language, source,
                              context_docs, question, review_type):
        # Step 1: Format context (same for all)
        context = format_code_context(context_docs)

        # Step 2: Get template (varies by type)
        prompt_template = get_review_prompt_template(review_type)

        # Step 3: Format prompt (same for all)
        prompt = prompt_template.format(...)

        # Step 4: Call LLM (same for all)
        return self.llm(prompt)

    # Concrete methods use template
    def review_code_with_context(self, ...):
        return self._review_with_template(..., "comprehensive")

    def detect_bugs(self, ...):
        return self._review_with_template(..., "bug_detection")
```

**Benefits:**
- Common structure extracted
- Variation isolated to templates
- Easy to add new review types

### 4.4 Facade Pattern

**Used in:** FileManager as facade

```python
# FileManager hides complexity of multi-step pipeline
class FileManager:
    def uploadFile(self):
        # Simple interface
        # Hides: validation → extraction → chunking → embedding → storage
        ...
```

**User sees:**
```python
POST /upload/file → Simple endpoint
```

**Actually happens:**
```
FileManager.uploadFile()
    → DocumentProcessor.pdfToText()
        → DocumentProcessor.chunkDocument()
            → VectorStore.store_chunks()
                → Embeddings.embed()
                    → ChromaDB.add()
```

**Benefits:**
- Simplified API
- Hides implementation complexity
- Single point of control

### 4.5 Repository Pattern

**Used in:** VectorStoreAndEmbedding

```python
class VectorStoreAndEmbedding:
    def __init__(self):
        self.vectorstore = Chroma(...)  # Database connection

    def store_chunks(self, chunks):
        # Abstract storage implementation
        self.vectorstore.add_documents(chunks)

    def search(self, query, k=5):
        # Abstract search implementation
        return self.vectorstore.max_marginal_relevance_search(query, k)
```

**Benefits:**
- Abstracts data access
- Could swap ChromaDB for Pinecone/Weaviate/Qdrant
- Business logic doesn't depend on specific DB

### 4.6 SOLID Principles

**Single Responsibility Principle:**
- DocumentProcessor: Only document processing
- CodeParser: Only code parsing
- LLMManager: Only LLM operations
- ChatSession: Only session management

**Open/Closed Principle:**
- Open for extension: Can add new languages to CodeParser
- Closed for modification: Don't change existing chunking logic

**Dependency Inversion Principle:**
- High-level modules (FileManager) depend on abstractions
- Low-level modules (DocumentProcessor) implement abstractions

---

## 5. Configuration & Setup

### 5.1 Environment Variables

**Optional .env file:**

```bash
# LLM Configuration
OLLAMA_MODEL=llama3.2
OLLAMA_TEMPERATURE=0.7
OLLAMA_BASE_URL=http://localhost:11434

# Embedding Model
EMBEDDING_MODEL=nomic-embed-text

# Vector Database
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=my_documents

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
CODE_CHUNK_SIZE=2000

# Session Configuration
SESSION_CLEANUP_HOURS=24

# Server Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5001
FLASK_DEBUG=True
```

**Loading in code:**

```python
from dotenv import load_dotenv
import os

load_dotenv()

model_name = os.getenv('OLLAMA_MODEL', 'llama3.2')
temperature = float(os.getenv('OLLAMA_TEMPERATURE', '0.7'))
```

### 5.2 System Requirements

**Python Version:**
- Python 3.8+ required
- Python 3.10+ recommended

**Dependencies:**
```bash
pip install -r requirements.txt
```

**External Services:**

1. **Ollama (Required)**
   ```bash
   # Install Ollama
   curl https://ollama.ai/install.sh | sh

   # Pull models
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. **FFmpeg (Optional - for YouTube)**
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg

   # macOS
   brew install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/
   ```

**Hardware Recommendations:**
- CPU: 4+ cores
- RAM: 8GB minimum, 16GB recommended
- Disk: 10GB for models + data
- GPU: Optional (speeds up Ollama)

### 5.3 Database Initialization

**ChromaDB Auto-Setup:**

```python
# First run creates database
vector_store = VectorStoreAndEmbedding()
# Creates ./chroma_db/ directory automatically
```

**Database Structure:**
```
chroma_db/
├── chroma.sqlite3          # SQLite database
└── index/                  # HNSW index files
```

**Collection Schema:**
```
Collection: "my_documents"
├── Vectors (768-dim)
├── Metadata
│   ├── source: str
│   ├── type: str
│   ├── language: str (for code)
│   ├── content_type: str
│   ├── structure_type: str (for code)
│   ├── structure_name: str (for code)
│   ├── start_line: int (for code)
│   └── end_line: int (for code)
└── Documents (text content)
```

### 5.4 Server Startup

**Development Mode:**

```bash
python main.py
```

Output:
```
🌐 Server starting on http://0.0.0.0:5001
 * Serving Flask app 'main'
 * Debug mode: on
 * Running on http://0.0.0.0:5001
```

**Production Mode:**

```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 main:app

# Using uWSGI
pip install uwsgi
uwsgi --http :5001 --wsgi-file main.py --callable app --processes 4
```

**Docker Deployment:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001
CMD ["python", "main.py"]
```

---

## 6. Extension Guide

### 6.1 Adding a New Language

**Step 1:** Add to LANGUAGE_EXTENSIONS

```python
# code_parser.py
LANGUAGE_EXTENSIONS = {
    '.go': 'go',  # Add Go support
    # ...
}
```

**Step 2:** Create extraction function

```python
def extract_go_functions(self, code: str):
    structures = []

    # Match Go function definitions
    pattern = r'^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\([^)]*\)'

    for match in re.finditer(pattern, code, re.MULTILINE):
        structures.append({
            'type': 'function',
            'name': match.group(1),
            'start': match.start()
        })

    return structures
```

**Step 3:** Add to chunk_code()

```python
def chunk_code(self, code, filename):
    language = self.detect_language(filename)

    if language == 'python':
        structures = self.extract_python_functions(code)
    elif language == 'go':
        structures = self.extract_go_functions(code)  # Add here
    # ...
```

### 6.2 Adding a New Review Type

**Step 1:** Create prompt template

```python
# code_review_prompts.py
ARCHITECTURE_REVIEW_PROMPT = """
You are a software architect. Review this code for architectural concerns:

**Code:**
```{language}
{code}
```

Analyze:
- Architectural patterns used
- Component coupling
- Scalability concerns
- Maintainability

Provide architectural recommendations.
"""
```

**Step 2:** Add to get_review_prompt_template()

```python
def get_review_prompt_template(review_type: str):
    prompts = {
        # ...
        "architecture": ARCHITECTURE_REVIEW_PROMPT,  # Add here
    }
```

**Step 3:** Create endpoint

```python
# main.py
@app.route('/review/architecture', methods=['POST'])
def architecture_review():
    # ... file handling ...

    review = llm_manager.review_code_with_context(
        code=code_content,
        language=language,
        source=filename,
        context_documents=context_docs,
        question="Analyze architectural design.",
        review_type="architecture"  # Use new type
    )

    return jsonify({'review': review})
```

### 6.3 Adding a New Vector Database

**Step 1:** Create new store class

```python
# vector_store_pinecone.py
import pinecone
from langchain.vectorstores import Pinecone

class PineconeVectorStore:
    def __init__(self, api_key, environment, index_name):
        pinecone.init(api_key=api_key, environment=environment)

        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings
        )

    def store_chunks(self, chunks):
        self.vectorstore.add_documents(chunks)
        return {'count': len(chunks)}

    def search(self, query, k=5):
        return self.vectorstore.similarity_search(query, k=k)
```

**Step 2:** Update main.py

```python
# main.py
# Choose vector store
USE_PINECONE = os.getenv('USE_PINECONE', 'false').lower() == 'true'

if USE_PINECONE:
    from src.modules.vector_store_pinecone import PineconeVectorStore
    vector_store = PineconeVectorStore(...)
else:
    vector_store = vector_store_module.VectorStoreAndEmbedding()
```

### 6.4 Adding File Type Support

**Step 1:** Install parser library

```bash
pip install docx2txt  # For .docx files
```

**Step 2:** Add extraction method

```python
# document_processor.py
import docx2txt

def docxToText(self, file_path, filename):
    """Extract text from .docx files"""
    text = docx2txt.process(file_path)
    return text
```

**Step 3:** Update uploadFile()

```python
# file_manager.py
elif filename.endswith('.docx'):
    content = self.document_processor.docxToText(destination, filename)
```

### 6.5 Adding Analytics

**Step 1:** Create analytics module

```python
# src/modules/analytics.py
from datetime import datetime
import json

class Analytics:
    def __init__(self):
        self.events = []

    def log_event(self, event_type, metadata):
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'metadata': metadata
        }
        self.events.append(event)

        # Optionally write to file
        with open('analytics.jsonl', 'a') as f:
            f.write(json.dumps(event) + '\n')

    def get_stats(self):
        return {
            'total_events': len(self.events),
            'uploads': sum(1 for e in self.events if e['type'] == 'upload'),
            'reviews': sum(1 for e in self.events if e['type'] == 'review'),
            'chats': sum(1 for e in self.events if e['type'] == 'chat')
        }
```

**Step 2:** Integrate in endpoints

```python
# main.py
analytics = Analytics()

@app.route('/upload/file', methods=['POST'])
def upload_file():
    result = file_manager.uploadFile()

    # Log event
    analytics.log_event('upload', {
        'filename': result.get('filename'),
        'file_type': result.get('file_type'),
        'chunks': result.get('chunks_created')
    })

    return jsonify(result)
```

---

## 7. Troubleshooting

### 7.1 Common Issues

**Issue:** "Ollama connection refused"

**Solution:**
```bash
# Check if Ollama is running
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

**Issue:** "Model not found"

**Solution:**
```bash
# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text

# List installed models
ollama list
```

**Issue:** "ChromaDB permission denied"

**Solution:**
```bash
# Check directory permissions
ls -la chroma_db/

# Fix permissions
chmod -R 755 chroma_db/
```

**Issue:** "Out of memory during embedding"

**Solution:**
```python
# Reduce chunk size
document_processor = DocumentProcessor()
document_processor.text_splitter.chunk_size = 500  # Smaller chunks
```

**Issue:** "Slow review responses"

**Solution:**
```python
# Use faster model
llm_manager = LLMManager(
    model_name="llama3.2:8b",  # Smaller model
    temperature=0.5
)
```

---

## 8. Performance Optimization

### 8.1 Caching Embeddings

```python
# Cache embeddings to avoid regeneration
from functools import lru_cache

class CachedEmbeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.cache = {}

    def embed_query(self, text):
        if text in self.cache:
            return self.cache[text]

        result = self.embeddings.embed_query(text)
        self.cache[text] = result
        return result
```

### 8.2 Batch Processing

```python
# Process multiple files at once
def batch_upload(files):
    chunks = []

    for file in files:
        content = extract_content(file)
        file_chunks = chunk_content(content)
        chunks.extend(file_chunks)

    # Single embedding call for all chunks
    vector_store.store_chunks(chunks)
```

### 8.3 Async Processing

```python
# Use async for concurrent operations
import asyncio

async def process_file_async(file):
    # Process in background
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, file_manager.uploadFile)
    return result
```

---

## 9. Security Considerations

### 9.1 Input Validation

```python
# Validate file uploads
ALLOWED_EXTENSIONS = {'.py', '.js', '.pdf', '.txt', '.md'}

def allowed_file(filename):
    return '.' in filename and \
           '.' + filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload/file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
```

### 9.2 Rate Limiting

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/review/comprehensive', methods=['POST'])
@limiter.limit("10 per hour")  # Limit expensive operations
def comprehensive_review():
    # ...
```

### 9.3 Authentication

```python
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')

        if api_key != os.getenv('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401

        return f(*args, **kwargs)
    return decorated_function

@app.route('/upload/file', methods=['POST'])
@require_api_key
def upload_file():
    # ...
```

---

## 10. Testing

### 10.1 Unit Tests

```python
# test_code_parser.py
import pytest
from src.modules.code_parser import CodeParser

def test_detect_language():
    parser = CodeParser()

    assert parser.detect_language('app.py') == 'python'
    assert parser.detect_language('main.js') == 'javascript'
    assert parser.detect_language('unknown.xyz') == 'unknown'

def test_extract_python_functions():
    parser = CodeParser()

    code = '''
def hello():
    print("Hello")

def world():
    print("World")
'''

    structures = parser.extract_python_functions(code)

    assert len(structures) == 2
    assert structures[0]['name'] == 'hello'
    assert structures[1]['name'] == 'world'
```

### 10.2 Integration Tests

```python
# test_api.py
import pytest
from main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    response = client.get('/health')

    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_upload_file(client):
    data = {'file': (BytesIO(b'Test content'), 'test.txt')}

    response = client.post('/upload/file', data=data)

    assert response.status_code == 200
    assert response.json['success'] == True
```

---

## Conclusion

This technical documentation provides a comprehensive explanation of every component in the dual-mode RAG chatbot system. The architecture is designed for:

- **Modularity:** Easy to extend and modify
- **Scalability:** Can handle growing data and users
- **Maintainability:** Clear separation of concerns
- **Flexibility:** Support for multiple file types and use cases

For questions or contributions, refer to the GitHub repository or contact the development team.
