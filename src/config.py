"""Central configuration. Everything is overridable through .env or the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

try:  # python-dotenv is optional at import time so tests can run bare
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

#: Exact wording the bot must use when the notes do not answer the question.
NOT_FOUND_MESSAGE = "Not found in your notes"

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(ENV_FILE, override=False)
        _ENV_LOADED = True


def _str(key: str, default: str) -> str:
    raw = os.getenv(key)
    return default if raw is None or not raw.strip() else raw.strip()


def _opt_str(key: str) -> str | None:
    raw = os.getenv(key)
    return raw.strip() if raw and raw.strip() else None


def _int(key: str, default: int) -> int:
    try:
        return int(_str(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(_str(key, str(default)))
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    return _str(key, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


def _path(key: str, default: Path) -> Path:
    raw = _opt_str(key)
    if raw is None:
        return default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass(frozen=True)
class Settings:
    # --- storage ---
    pdf_dir: Path
    index_dir: Path
    collection: str

    # --- ingestion ---
    chunk_tokens: int
    chunk_overlap: int
    min_chunk_chars: int

    # --- embeddings ---
    embed_model: str
    embed_backend: str  # sentence-transformers | hash
    embed_batch_size: int

    # --- retrieval ---
    top_k: int
    fetch_k: int
    reranker: str  # none | lexical | cross-encoder
    cross_encoder_model: str

    # --- library tiers ---
    default_visibility: str
    injection_enforce: bool

    # --- llm ---
    llm_provider: str  # anthropic | ollama | extractive
    anthropic_model: str
    anthropic_api_key: str | None
    ollama_host: str
    ollama_model: str
    max_tokens: int
    temperature: float

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def with_overrides(self, **kwargs: Any) -> "Settings":
        return replace(self, **kwargs)

    def describe(self) -> dict[str, Any]:
        return {
            "pdf_dir": str(self.pdf_dir),
            "index_dir": str(self.index_dir),
            "collection": self.collection,
            "chunking": f"{self.chunk_tokens} tokens / {self.chunk_overlap} overlap",
            "embed_model": self.embed_model if self.embed_backend != "hash" else "hash (offline)",
            "retrieval": f"top_k={self.top_k} fetch_k={self.fetch_k} reranker={self.reranker}",
            "llm_provider": self.llm_provider,
            "llm_model": self.anthropic_model
            if self.llm_provider == "anthropic"
            else (self.ollama_model if self.llm_provider == "ollama" else "-"),
        }


def load_settings(**overrides: Any) -> Settings:
    """Build Settings from the environment, then apply explicit overrides."""
    _ensure_env_loaded()
    settings = Settings(
        pdf_dir=_path("PDF_DIR", PROJECT_ROOT / "data" / "pdfs"),
        index_dir=_path("INDEX_DIR", PROJECT_ROOT / "data" / "index"),
        collection=_str("CHROMA_COLLECTION", "vtu_notes"),
        chunk_tokens=_int("CHUNK_TOKENS", 500),
        chunk_overlap=_int("CHUNK_OVERLAP", 80),
        min_chunk_chars=_int("MIN_CHUNK_CHARS", 40),
        embed_model=_str("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        embed_backend=_str("EMBED_BACKEND", "sentence-transformers").lower(),
        embed_batch_size=_int("EMBED_BATCH_SIZE", 32),
        default_visibility=_str("DEFAULT_VISIBILITY", "curated"),
        injection_enforce=_bool("INJECTION_ENFORCE", False),
        top_k=_int("TOP_K", 5),
        fetch_k=_int("FETCH_K", 20),
        reranker=_str("RERANKER", "lexical").lower(),
        cross_encoder_model=_str("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base"),
        llm_provider=_str("LLM_PROVIDER", "anthropic").lower(),
        anthropic_model=_str("ANTHROPIC_MODEL", "claude-sonnet-5"),
        anthropic_api_key=_opt_str("ANTHROPIC_API_KEY"),
        ollama_host=_str("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        ollama_model=_str("OLLAMA_MODEL", "llama3.1:8b"),
        max_tokens=_int("LLM_MAX_TOKENS", 1200),
        temperature=_float("LLM_TEMPERATURE", 0.0),
    )
    return settings.with_overrides(**overrides) if overrides else settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Use load_settings() in tests."""
    return load_settings()
