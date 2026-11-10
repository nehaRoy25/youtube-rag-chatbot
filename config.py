"""Application configuration and environment management."""

from pathlib import Path
import os
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
STYLE_PATH = ASSETS_DIR / "style.css"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env (overrides nothing if already exported)
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Application metadata
APP_TITLE = "YouTube Video RAG Assistant"
APP_SUBTITLE = "Ask grounded questions about any YouTube video with timestamped citations"
APP_ICON = "🎬"

# Groq LLM configuration
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()

AVAILABLE_GROQ_MODELS: list[str] = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
DEFAULT_GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 2048

# Embedding model configuration
AVAILABLE_EMBEDDING_MODELS: list[str] = [
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
]
DEFAULT_EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunking & retrieval defaults
DEFAULT_CHUNK_SIZE: int = 800
MIN_CHUNK_SIZE: int = 250
MAX_CHUNK_SIZE: int = 2000

DEFAULT_CHUNK_OVERLAP: int = 150
MIN_CHUNK_OVERLAP: int = 0
MAX_CHUNK_OVERLAP: int = 500

DEFAULT_TOP_K: int = 4
MIN_TOP_K: int = 3
MAX_TOP_K: int = 5


def is_groq_configured() -> bool:
    """Return True if a non-placeholder Groq API key is configured."""
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(
        key
        and key != "your_groq_api_key_here"
        and len(key) > 10
    )
