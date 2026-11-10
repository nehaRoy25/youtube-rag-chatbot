"""Transcript text chunking with LangChain RecursiveCharacterTextSplitter and timestamp mapping."""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.transcript import TranscriptResult, TranscriptSegment
from modules.utils import build_youtube_timestamp_url, count_words, format_timestamp


@dataclass
class TextChunk:
    """A single chunk of transcript text paired with exact source timestamps."""

    chunk_id: str
    chunk_index: int
    video_id: str
    content: str
    char_count: int
    word_count: int
    start_index: int
    start_time: Optional[float]
    end_time: Optional[float]
    timestamp: str
    timestamp_range: str
    youtube_url: str


@dataclass
class ChunkingStats:
    """Summary statistics for generated chunks."""

    total_chunks: int
    chunk_size_config: int
    chunk_overlap_config: int
    avg_char_count: float
    min_char_count: int
    max_char_count: int
    total_words: int


def _find_overlapping_segments(
    segments: list[TranscriptSegment],
    segment_ends: list[int],
    char_start: int,
    char_end: int,
) -> list[TranscriptSegment]:
    """Find transcript segments overlapping the character range [char_start, char_end)."""
    if not segments:
        return []

    idx = bisect_right(segment_ends, char_start)
    overlapping: list[TranscriptSegment] = []

    while idx < len(segments) and segments[idx].char_start < char_end:
        if segments[idx].char_end > char_start:
            overlapping.append(segments[idx])
        idx += 1

    if not overlapping:
        nearest_idx = min(max(0, bisect_right(segment_ends, char_start)), len(segments) - 1)
        overlapping.append(segments[nearest_idx])

    return overlapping


def chunk_transcript(
    transcript: TranscriptResult,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[list[TextChunk], ChunkingStats]:
    """Split a transcript into overlapping chunks while preserving accurate timestamps."""
    if not transcript.full_text.strip():
        raise ValueError("Cannot chunk an empty transcript.")
    if chunk_size < 100:
        raise ValueError("Chunk size must be at least 100 characters.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    documents = splitter.create_documents([transcript.full_text])
    total_docs = len(documents)
    segment_ends = [seg.char_end for seg in transcript.segments]

    chunks: list[TextChunk] = []
    search_cursor = 0

    for idx, doc in enumerate(documents):
        content = doc.page_content.strip()
        if not content:
            continue

        start_index = doc.metadata.get("start_index", -1)
        if start_index is None or start_index < 0:
            found_at = transcript.full_text.find(content, max(0, search_cursor))
            if found_at == -1:
                found_at = transcript.full_text.find(content)
            start_index = max(0, found_at)
        else:
            # Adjust if stripping leading whitespace shifted content start
            raw_content = doc.page_content
            leading_ws = len(raw_content) - len(raw_content.lstrip())
            start_index += leading_ws

        char_end = start_index + len(content)
        search_cursor = max(0, start_index + 1)

        overlapping = _find_overlapping_segments(
            transcript.segments, segment_ends, start_index, char_end
        )
        start_time = overlapping[0].start if overlapping else None
        end_time = overlapping[-1].end if overlapping else None

        timestamp = format_timestamp(start_time)
        timestamp_range = (
            f"{format_timestamp(start_time)} – {format_timestamp(end_time)}"
            if start_time is not None and end_time is not None
            else "N/A"
        )
        youtube_url = build_youtube_timestamp_url(transcript.video_id, start_time)

        chunk_num = len(chunks) + 1
        chunks.append(
            TextChunk(
                chunk_id=f"chunk_{chunk_num:03d}",
                chunk_index=chunk_num - 1,
                video_id=transcript.video_id,
                content=content,
                char_count=len(content),
                word_count=count_words(content),
                start_index=start_index,
                start_time=start_time,
                end_time=end_time,
                timestamp=timestamp,
                timestamp_range=timestamp_range,
                youtube_url=youtube_url,
            )
        )

        if progress_callback is not None:
            progress_callback(idx + 1, total_docs)

    if not chunks:
        raise ValueError("No valid chunks were generated from the transcript.")

    char_counts = [c.char_count for c in chunks]
    stats = ChunkingStats(
        total_chunks=len(chunks),
        chunk_size_config=chunk_size,
        chunk_overlap_config=chunk_overlap,
        avg_char_count=round(sum(char_counts) / len(char_counts), 1),
        min_char_count=min(char_counts),
        max_char_count=max(char_counts),
        total_words=sum(c.word_count for c in chunks),
    )
    return chunks, stats
