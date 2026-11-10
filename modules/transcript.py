"""YouTube transcript extraction and normalization module."""

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from modules.utils import count_words, format_timestamp

try:
    from youtube_transcript_api._errors import (
        AgeRestricted,
        InvalidVideoId,
        IpBlocked,
        PoTokenRequired,
        RequestBlocked,
        VideoUnplayable,
        YouTubeRequestFailed,
    )
except ImportError:  # Fallback for older youtube-transcript-api versions
    AgeRestricted = CouldNotRetrieveTranscript
    InvalidVideoId = CouldNotRetrieveTranscript
    IpBlocked = CouldNotRetrieveTranscript
    PoTokenRequired = CouldNotRetrieveTranscript
    RequestBlocked = CouldNotRetrieveTranscript
    VideoUnplayable = VideoUnavailable
    YouTubeRequestFailed = CouldNotRetrieveTranscript


PREFERRED_ENGLISH_CODES = ("en", "en-US", "en-GB", "en-AU", "en-CA", "en-IN")


class TranscriptUnavailableError(Exception):
    """Raised when a video transcript cannot be fetched or is empty."""


@dataclass
class TranscriptSegment:
    """Single spoken caption segment with timing and character span in full_text."""

    text: str
    start: float
    duration: float
    char_start: int
    char_end: int

    @property
    def end(self) -> float:
        return self.start + max(0.0, self.duration)

    @property
    def formatted_start(self) -> str:
        return format_timestamp(self.start)


@dataclass
class TranscriptResult:
    """Complete extracted transcript with segment timing metadata."""

    video_id: str
    segments: list[TranscriptSegment]
    full_text: str
    word_count: int
    char_count: int
    language: str
    language_code: str
    is_generated: bool
    duration_seconds: float

    @property
    def formatted_duration(self) -> str:
        return format_timestamp(self.duration_seconds)


def _select_transcript(transcript_list: Any) -> tuple[Any, bool]:
    """Choose the best transcript from a TranscriptList.

    Returns (transcript_obj, was_translated).
    Priority:
      1. Manually created English transcript
      2. Auto-generated English transcript
      3. Any manually created transcript (translated to English if translatable)
      4. First available transcript (translated to English if translatable)
    """
    # 1. Manual English
    try:
        return transcript_list.find_manually_created_transcript(PREFERRED_ENGLISH_CODES), False
    except NoTranscriptFound:
        pass

    # 2. Generated English
    try:
        return transcript_list.find_generated_transcript(PREFERRED_ENGLISH_CODES), False
    except NoTranscriptFound:
        pass

    # 3. Fallback to any available transcript
    available = list(transcript_list)
    if not available:
        raise TranscriptUnavailableError("No transcripts or captions are available for this video.")

    manual_first = sorted(available, key=lambda t: getattr(t, "is_generated", True))
    chosen = manual_first[0]

    if getattr(chosen, "is_translatable", False) and not str(
        getattr(chosen, "language_code", "")
    ).startswith("en"):
        try:
            return chosen.translate("en"), True
        except Exception:
            return chosen, False

    return chosen, False


def _normalize_raw_snippets(raw_snippets: Iterable[Any]) -> tuple[list[TranscriptSegment], str]:
    """Convert raw transcript snippet dicts/objects into cleaned TranscriptSegments and full_text."""
    segments: list[TranscriptSegment] = []
    text_parts: list[str] = []
    cursor = 0

    for item in raw_snippets:
        if isinstance(item, dict):
            raw_text = item.get("text", "")
            start = float(item.get("start", 0.0) or 0.0)
            duration = float(item.get("duration", 0.0) or 0.0)
        else:
            raw_text = getattr(item, "text", "")
            start = float(getattr(item, "start", 0.0) or 0.0)
            duration = float(getattr(item, "duration", 0.0) or 0.0)

        cleaned_text = " ".join(str(raw_text).replace("\n", " ").split())
        if not cleaned_text:
            continue

        if text_parts:
            cursor += 1  # account for single space separator when joining

        char_start = cursor
        char_end = char_start + len(cleaned_text)
        cursor = char_end

        text_parts.append(cleaned_text)
        segments.append(
            TranscriptSegment(
                text=cleaned_text,
                start=max(0.0, start),
                duration=max(0.0, duration),
                char_start=char_start,
                char_end=char_end,
            )
        )

    full_text = " ".join(text_parts)
    return segments, full_text


def extract_transcript(video_id: str) -> TranscriptResult:
    """Extract and normalize the transcript for a YouTube video ID."""
    if not video_id:
        raise TranscriptUnavailableError("Invalid YouTube video ID.")

    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "list"):
            transcript_list = api.list(video_id)
        else:  # Legacy youtube-transcript-api < 1.0.0 compatibility
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        selected_transcript, was_translated = _select_transcript(transcript_list)
        fetched = selected_transcript.fetch()

        if hasattr(fetched, "to_raw_data"):
            raw_snippets = fetched.to_raw_data()
        else:
            raw_snippets = list(fetched)

        language = getattr(selected_transcript, "language", "English")
        language_code = getattr(selected_transcript, "language_code", "en")
        is_generated = bool(getattr(selected_transcript, "is_generated", False))
        if was_translated:
            language = f"English (Translated from {language})"
            language_code = "en"

    except TranscriptUnavailableError:
        raise
    except TranscriptsDisabled as exc:
        raise TranscriptUnavailableError(
            "Transcripts and captions are disabled for this YouTube video."
        ) from exc
    except NoTranscriptFound as exc:
        raise TranscriptUnavailableError(
            "No spoken transcript or closed captions could be found for this video."
        ) from exc
    except AgeRestricted as exc:
        raise TranscriptUnavailableError(
            "This video is age-restricted, so YouTube prevents public transcript extraction."
        ) from exc
    except (VideoUnavailable, VideoUnplayable, InvalidVideoId) as exc:
        raise TranscriptUnavailableError(
            "This YouTube video is unavailable, private, or does not exist."
        ) from exc
    except (IpBlocked, RequestBlocked, PoTokenRequired) as exc:
        raise TranscriptUnavailableError(
            "YouTube temporarily blocked transcript requests from this network/IP. "
            "Please try again later or test another network."
        ) from exc
    except (YouTubeRequestFailed, CouldNotRetrieveTranscript) as exc:
        raise TranscriptUnavailableError(
            f"Unable to retrieve transcript from YouTube: {exc}"
        ) from exc
    except Exception as exc:
        raise TranscriptUnavailableError(
            f"Unexpected error while extracting transcript: {exc}"
        ) from exc

    segments, full_text = _normalize_raw_snippets(raw_snippets)
    if not segments or not full_text.strip():
        raise TranscriptUnavailableError(
            "The extracted transcript is empty or contains no readable spoken words."
        )

    last_seg = segments[-1]
    duration_seconds = last_seg.end

    return TranscriptResult(
        video_id=video_id,
        segments=segments,
        full_text=full_text,
        word_count=count_words(full_text),
        char_count=len(full_text),
        language=language,
        language_code=language_code,
        is_generated=is_generated,
        duration_seconds=duration_seconds,
    )
