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
REVIEW_LLM_TEMPERATURE = float(os.getenv("REVIEW_LLM_TEMPERATURE", 0.2))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ===== Vector store =====
CHROMA_PERSIST_DIR = _path("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "my_documents")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 5))

# ===== Uploads =====
UPLOAD_DIR = _path("UPLOAD_DIR", "./uploads")
MEDIA_DIR = _path("MEDIA_DIR", "./uploads/media")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", 20)) * 1024 * 1024
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# ===== Code review service (Node.js) =====
REVIEW_SERVICE_URL = os.getenv("REVIEW_SERVICE_URL", "http://localhost:4000").rstrip("/")
REVIEW_TIMEOUT_SECONDS = int(os.getenv("REVIEW_TIMEOUT_SECONDS", 60))

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
for _d in (UPLOAD_DIR, MEDIA_DIR, CHROMA_PERSIST_DIR):
    _d.mkdir(parents=True, exist_ok=True)