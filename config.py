"""Application configuration using Streamlit Secrets with environment fallback."""

from collections.abc import Mapping
import os
from pathlib import Path

from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
STYLE_PATH = ASSETS_DIR / "style.css"

# Ensure data directory exists when filesystem is writable
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

# Optional local .env fallback for non-Streamlit CLI scripts
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

# Application metadata
APP_TITLE = "YouTube Video RAG Assistant"
APP_SUBTITLE = "Ask grounded questions about any YouTube video with timestamped citations"
APP_ICON = "🎬"

AVAILABLE_GROQ_MODELS: list[str] = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 2048

# Embedding model configuration
AVAILABLE_EMBEDDING_MODELS: list[str] = [
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
]

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


def _read_streamlit_secret(key: str) -> str:
    """Safely read a key from st.secrets without failing if secrets.toml is absent."""
    try:
        import streamlit as st

        if key in st.secrets:
            val = st.secrets[key]
            if val is not None:
                return str(val).strip()

        if key == "GROQ_API_KEY" and "groq" in st.secrets:
            groq_sec = st.secrets["groq"]
            if isinstance(groq_sec, Mapping):
                for subkey in ("api_key", "GROQ_API_KEY"):
                    if subkey in groq_sec and groq_sec[subkey]:
                        return str(groq_sec[subkey]).strip()
    except Exception:
        pass
    return ""


def get_secret(key: str, default: str = "") -> str:
    """Read configuration value from Streamlit Secrets first, then environment variables."""
    secret_val = _read_streamlit_secret(key)
    if secret_val:
        return secret_val

    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    env_val = os.getenv(key, "").strip()
    if env_val:
        return env_val

    return default


def get_groq_api_key() -> str:
    """Return the configured Groq API key from Streamlit Secrets or environment."""
    return get_secret("GROQ_API_KEY", "")


def get_groq_model() -> str:
    """Return the preferred Groq model from Streamlit Secrets or default."""
    return get_secret("GROQ_MODEL", "qwen/qwen3.8-27b")


def get_embedding_model() -> str:
    """Return the preferred SentenceTransformer model from Streamlit Secrets or default."""
    return get_secret("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


DEFAULT_GROQ_MODEL: str = get_groq_model()
DEFAULT_EMBEDDING_MODEL: str = get_embedding_model()


def _is_valid_key(key: str) -> bool:
    return bool(key and key != "your_groq_api_key_here" and len(key) > 10)


def get_secret_source() -> str:
    """Return where the active Groq API key was loaded from (never returns the key itself)."""
    if _is_valid_key(_read_streamlit_secret("GROQ_API_KEY")):
        return "Streamlit Secrets"
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    if _is_valid_key(os.getenv("GROQ_API_KEY", "").strip()):
        return "Environment"
    return "Missing"


def is_groq_configured() -> bool:
    """Return True if a non-placeholder Groq API key is configured."""
    return _is_valid_key(get_groq_api_key())
