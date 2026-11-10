"""Semantic retrieval module bridging query embedding and FAISS similarity search."""

from dataclasses import dataclass

from modules.chunking import TextChunk
from modules.embeddings import EmbeddingManager
from modules.vector_store import FAISSVectorStore


class RetrievalError(Exception):
    """Raised when similarity retrieval cannot be performed."""


@dataclass
class RetrievedChunk:
    """Single retrieved transcript chunk paired with similarity score and rank."""

    rank: int
    chunk: TextChunk
    similarity_score: float
    similarity_pct: float


@dataclass
class RetrievalResult:
    """Full retrieval output for a user question."""

    query: str
    top_k: int
    query_dimension: int
    results: list[RetrievedChunk]


class Retriever:
    """Retrieves top-k relevant transcript chunks for user questions."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: FAISSVectorStore,
    ) -> None:
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        """Embed a user question and retrieve top-k matching chunks from FAISS."""
        if not query or not query.strip():
            raise RetrievalError("Please enter a non-empty question.")
        if not self.vector_store.is_ready:
            raise RetrievalError(
                "No processed video found in the vector database. "
                "Please process a YouTube video first."
            )

        query_vec = self.embedding_manager.embed_query(query)
        raw_matches = self.vector_store.search(query_vec, top_k=top_k)

        retrieved: list[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(raw_matches, start=1):
            bounded_pct = round(max(0.0, min(100.0, score * 100.0)), 1)
            retrieved.append(
                RetrievedChunk(
                    rank=rank,
                    chunk=chunk,
                    similarity_score=round(float(score), 4),
                    similarity_pct=bounded_pct,
                )
            )

        return RetrievalResult(
            query=query.strip(),
            top_k=top_k,
            query_dimension=self.embedding_manager.dimension,
            results=retrieved,
        )

    @staticmethod
    def format_context_for_llm(retrieval_result: RetrievalResult) -> str:
        """Format retrieved chunks into a grounded, citation-friendly prompt block."""
        if not retrieval_result.results:
            return "No transcript excerpts were retrieved."

        sections: list[str] = []
        for item in retrieval_result.results:
            c = item.chunk
            sections.append(
                f"[Source: {c.chunk_id} | Timestamp: {c.timestamp_range} | "
                f"Similarity: {item.similarity_score:.4f}]\n{c.content}"
            )
        return "\n\n---\n\n".join(sections)
