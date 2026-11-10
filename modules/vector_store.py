"""FAISS vector database wrapper storing chunk embeddings and metadata."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Optional

import faiss
import numpy as np

from modules.chunking import TextChunk


class VectorStoreError(Exception):
    """Raised when FAISS vector indexing, searching, or persistence fails."""


@dataclass
class VectorStoreStats:
    """Live metrics and status of the FAISS vector store."""

    total_vectors: int
    dimension: int
    index_type: str
    status: str
    video_id: str
    embedding_model: str
    created_at: str
    persisted_path: Optional[str] = None


class FAISSVectorStore:
    """In-memory and disk-persistable FAISS vector index with chunk metadata."""

    def __init__(
        self,
        dimension: int,
        video_id: str = "",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        if dimension <= 0:
            raise VectorStoreError("Embedding dimension must be positive.")

        self.dimension: int = int(dimension)
        self.video_id: str = video_id
        self.embedding_model: str = embedding_model
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.chunks: list[TextChunk] = []
        self.created_at: str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.persisted_path: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """Return True when at least one vector is indexed and aligned with chunk metadata."""
        return self.index.ntotal > 0 and self.index.ntotal == len(self.chunks)

    def add_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: np.ndarray,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Normalize vectors and store them alongside chunk metadata in FAISS."""
        if not chunks:
            raise VectorStoreError("No chunks supplied for vector storage.")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise VectorStoreError(
                f"Embeddings shape {embeddings.shape} does not match {len(chunks)} chunks."
            )
        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Expected vector dimension {self.dimension}, got {embeddings.shape[1]}."
            )

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        try:
            batch_size = max(1, min(64, len(chunks)))
            total = len(chunks)
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                self.index.add(vectors[start:end])
                self.chunks.extend(chunks[start:end])
                if progress_callback is not None:
                    progress_callback(end, total)
        except Exception as exc:
            raise VectorStoreError(f"Failed to add vectors to FAISS index: {exc}") from exc

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
    ) -> list[tuple[TextChunk, float]]:
        """Perform cosine similarity search against indexed chunk vectors."""
        if not self.is_ready:
            raise VectorStoreError("Vector store is empty. Process a video before searching.")
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than 0.")

        query_vec = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(
            1, self.dimension
        )
        faiss.normalize_L2(query_vec)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results: list[tuple[TextChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[int(idx)], float(score)))

        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def save(self, base_dir: Path) -> Path:
        """Persist FAISS index and chunk metadata to disk under data/<video_id>/."""
        if not self.is_ready:
            raise VectorStoreError("Cannot save an empty vector store.")

        target_dir = base_dir / (self.video_id or "default")
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "index.faiss"
        meta_file = target_dir / "metadata.json"

        faiss.write_index(self.index, str(index_file))
        payload = {
            "video_id": self.video_id,
            "dimension": self.dimension,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        meta_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.persisted_path = str(target_dir)
        return target_dir

    @classmethod
    def load(cls, directory: Path) -> "FAISSVectorStore":
        """Load a persisted FAISS index and chunk metadata from disk."""
        index_file = directory / "index.faiss"
        meta_file = directory / "metadata.json"
        if not index_file.exists() or not meta_file.exists():
            raise VectorStoreError(f"Saved FAISS index not found in {directory}.")

        payload = json.loads(meta_file.read_text(encoding="utf-8"))
        store = cls(
            dimension=int(payload["dimension"]),
            video_id=str(payload.get("video_id", "")),
            embedding_model=str(payload.get("embedding_model", "all-MiniLM-L6-v2")),
        )
        store.created_at = str(payload.get("created_at", store.created_at))
        store.index = faiss.read_index(str(index_file))
        store.chunks = [TextChunk(**chunk_dict) for chunk_dict in payload["chunks"]]
        store.persisted_path = str(directory)
        return store

    def clear(self) -> None:
        """Reset the FAISS index and remove stored chunk metadata."""
        self.index.reset()
        self.chunks.clear()
        self.persisted_path = None

    def get_stats(self) -> VectorStoreStats:
        """Return current FAISS index statistics."""
        return VectorStoreStats(
            total_vectors=int(self.index.ntotal),
            dimension=self.dimension,
            index_type="FAISS IndexFlatIP (Cosine Similarity)",
            status="Ready" if self.is_ready else "Empty",
            video_id=self.video_id,
            embedding_model=self.embedding_model,
            created_at=self.created_at,
            persisted_path=self.persisted_path,
        )
