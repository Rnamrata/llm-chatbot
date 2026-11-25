"""Central configuration. Values come from .env, with safe defaults."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Project root (llm-chatbot/), so paths work no matter where you run from
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")


def _path(name, default):
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else BASE_DIR / p


# ===== Flask server =====
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5001))
FLASK_DEBUG = _bool("FLASK_DEBUG", False)
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",") if o.strip()]

# ===== Ollama / LLM =====
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
# Tokens the chat model may read at once: RETRIEVAL_K passages plus the chat history
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", 8192))
# Ask Lens (code-review chat) has its own model: a coding model reads code better
# than the general chat model, and the review service already uses this one
REVIEW_LLM_MODEL = os.getenv("REVIEW_LLM_MODEL", "qwen2.5-coder:7b")
# Temperature 0 and a fixed seed: the same question about the same code gets the
# same answer, as the review service does, so a good answer stays good
REVIEW_LLM_TEMPERATURE = float(os.getenv("REVIEW_LLM_TEMPERATURE", 0))
REVIEW_LLM_SEED = int(os.getenv("REVIEW_LLM_SEED", 42))
# Tokens the review model may read at once. Ask Lens sends the whole numbered file and
# every finding, which does not fit Ollama's default.
REVIEW_NUM_CTX = int(os.getenv("REVIEW_NUM_CTX", 16384))
# A file longer than this (in characters) is searched instead of sent whole
REVIEW_FULL_CODE_CHARS = int(os.getenv("REVIEW_FULL_CODE_CHARS", 30000))
# Earlier questions and answers an Ask Lens prompt includes
REVIEW_HISTORY_MESSAGES = int(os.getenv("REVIEW_HISTORY_MESSAGES", 8))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# nomic-embed-text is trained to see what a text is for: stored passages start with
# "search_document: " and questions with "search_query: ". Without them the right
# passage ranks lower. Set both to "" for a model that does not use prefixes.
# Changing them means re-embedding what is stored (scripts/reembed.py).
EMBED_DOCUMENT_PREFIX = os.getenv("EMBED_DOCUMENT_PREFIX", "search_document: ")
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query: ")

# ===== Vector store =====
CHROMA_PERSIST_DIR = _path("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "my_documents")
# Passages a document chat reads per question. 5 left answers ranked just below the top out.
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 8))
# Chunks embedded per request to Ollama. A whole book in one request means one dropped
# connection loses the upload; smaller batches are retried on their own.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 64))
EMBED_ATTEMPTS = int(os.getenv("EMBED_ATTEMPTS", 3))
# ===== Uploads =====
UPLOAD_DIR = _path("UPLOAD_DIR", "./uploads")
MEDIA_DIR = _path("MEDIA_DIR", "./uploads/media")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", 20)) * 1024 * 1024
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# ===== Chat sessions =====
# Saved here so History and open chats survive a restart
SESSIONS_FILE = _path("SESSIONS_FILE", "./data/sessions.json")

# Source overviews ("summarize this"): built in the background when a source is added
OVERVIEW_ENABLED = _bool("OVERVIEW_ENABLED", True)
OVERVIEWS_FILE = SESSIONS_FILE.parent / "overviews.json"
# A source with no more text than this is summarized whole; a longer one from one passage per topic
OVERVIEW_WHOLE_CHARS = int(os.getenv("OVERVIEW_WHOLE_CHARS", 20000))
# Most text the model reads in one overview call (llama3.2 reads 8K tokens at once)
OVERVIEW_BATCH_CHARS = int(os.getenv("OVERVIEW_BATCH_CHARS", 20000))
# Most topic groups a long source is split into
OVERVIEW_TOPICS = int(os.getenv("OVERVIEW_TOPICS", 20))
# How long a chat waits for an overview still being built. Keep it under Lens's 180 s answer timeout.
OVERVIEW_WAIT_SECONDS = float(os.getenv("OVERVIEW_WAIT_SECONDS", 150))

# ===== Code review service (Node.js) =====
REVIEW_SERVICE_URL = os.getenv("REVIEW_SERVICE_URL", "http://localhost:4000").rstrip("/")
# Longer than the review service's own AI limit (LLM_TIMEOUT_MS, 90 s), so a slow AI review
# comes back with the linters' findings and a warning instead of failing here
REVIEW_TIMEOUT_SECONDS = int(os.getenv("REVIEW_TIMEOUT_SECONDS", 120))

# ===== Chunking =====
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
CODE_CHUNK_SIZE = 1500
CODE_CHUNK_OVERLAP = 150

# ===== File types =====
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}

# extension -> language (values match LangChain's `Language` enum,
# so the code splitter can use Language(value) directly)
CODE_LANGUAGES = {
    ".py": "python",
    ".js": "js",
    ".jsx": "js",
    ".ts": "ts",
    ".tsx": "ts",
    ".java": "java",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".rs": "rust",
    ".kt": "kotlin",
    ".swift": "swift",
}
CODE_EXTENSIONS = set(CODE_LANGUAGES)

# Make sure folders exist at startup
for _d in (UPLOAD_DIR, MEDIA_DIR, CHROMA_PERSIST_DIR, SESSIONS_FILE.parent):
    _d.mkdir(parents=True, exist_ok=True)