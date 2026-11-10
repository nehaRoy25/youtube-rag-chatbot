"""Sentence Transformer embedding generation module."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingError(Exception):
    """Raised when embedding model initialization or encoding fails."""


@dataclass
class EmbeddingSummary:
    """Metadata and sample representation for generated chunk embeddings."""

    model_name: str
    dimension: int
    total_embeddings: int
    sample_vector_preview: list[float]
    sample_vector_norm: float


@lru_cache(maxsize=3)
def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Cache loaded SentenceTransformer models in memory across queries and reruns."""
    return SentenceTransformer(model_name)


class EmbeddingManager:
    """Manages HuggingFace SentenceTransformer embeddings for chunks and queries."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        try:
            self.model = _load_sentence_transformer(model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{model_name}': {exc}"
            ) from exc

        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        if dim is None or dim <= 0:
            raise EmbeddingError(f"Could not determine embedding dimension for '{model_name}'.")
        self.dimension: int = int(dim)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """Encode a list of texts into L2-normalized float32 embeddings."""
        if not texts:
            raise EmbeddingError("Cannot generate embeddings for an empty text list.")

        total = len(texts)
        all_vectors: list[np.ndarray] = []

        try:
            for start in range(0, total, batch_size):
                batch = texts[start : start + batch_size]
                batch_vectors = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                all_vectors.append(np.asarray(batch_vectors, dtype=np.float32))
                if progress_callback is not None:
                    progress_callback(min(start + len(batch), total), total)
        except Exception as exc:
            raise EmbeddingError(f"Error during text embedding generation: {exc}") from exc

        embeddings = np.vstack(all_vectors).astype(np.float32)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single user question into a normalized (1, dim) float32 vector."""
        if not query or not query.strip():
            raise EmbeddingError("Query cannot be empty.")

        try:
            vector = self.model.encode(
                [query.strip()],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(vector, dtype=np.float32).reshape(1, self.dimension)
        except Exception as exc:
            raise EmbeddingError(f"Error encoding query into embedding: {exc}") from exc

    def build_summary(self, embeddings: np.ndarray, preview_dims: int = 12) -> EmbeddingSummary:
        """Build summary statistics and a sample vector preview for UI inspection."""
        if embeddings.size == 0:
            raise EmbeddingError("Cannot summarize empty embeddings array.")

        first_vec = embeddings[0]
        preview = [round(float(x), 6) for x in first_vec[:preview_dims]]
        norm = round(float(np.linalg.norm(first_vec)), 6)

        return EmbeddingSummary(
            model_name=self.model_name,
            dimension=self.dimension,
            total_embeddings=int(embeddings.shape[0]),
            sample_vector_preview=preview,
            sample_vector_norm=norm,
        )
