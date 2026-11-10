"""Utility functions for YouTube URL parsing, metadata lookup, timestamps, and text formatting."""

from dataclasses import dataclass
import html
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
import requests

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "them", "there", "these",
    "they", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your", "about", "into",
    "video", "youtube", "explain", "tell", "say", "said", "talk", "mentioned",
}


class InvalidYouTubeURLError(ValueError):
    """Raised when a provided string is not a valid YouTube video URL."""


@dataclass
class VideoMetadata:
    """Metadata extracted for a YouTube video."""

    video_id: str
    url: str
    title: str
    channel: str
    thumbnail_url: str


def extract_video_id(url: str) -> str:
    """Validate a YouTube URL and extract its 11-character video ID.

    Supports standard watch URLs, shortened youtu.be links, Shorts, Live,
    and embed URLs.
    """
    if not url or not isinstance(url, str):
        raise InvalidYouTubeURLError("Please enter a YouTube video URL.")

    cleaned = url.strip()
    if not cleaned:
        raise InvalidYouTubeURLError("Please enter a YouTube video URL.")

    if not re.match(r"^https?://", cleaned, flags=re.IGNORECASE):
        if any(
            cleaned.lower().startswith(prefix)
            for prefix in ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be")
        ):
            cleaned = f"https://{cleaned}"
        else:
            raise InvalidYouTubeURLError(
                "Invalid URL format. Please enter a full YouTube URL such as "
                "https://www.youtube.com/watch?v=..."
            )

    parsed = urlparse(cleaned)
    host = parsed.netloc.lower().split(":")[0]
    if host not in YOUTUBE_HOSTS:
        raise InvalidYouTubeURLError(
            "URL does not point to YouTube. Please provide a valid youtube.com or youtu.be link."
        )

    video_id: Optional[str] = None
    path_parts = [part for part in parsed.path.split("/") if part]

    if "youtu.be" in host:
        if path_parts:
            video_id = path_parts[0]
    else:
        query_params = parse_qs(parsed.query)
        if "v" in query_params and query_params["v"]:
            video_id = query_params["v"][0]
        elif len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live", "v"}:
            video_id = path_parts[1]

    if not video_id or not YOUTUBE_ID_PATTERN.match(video_id):
        raise InvalidYouTubeURLError(
            "Could not extract a valid 11-character YouTube video ID from that URL."
        )

    return video_id


def is_valid_youtube_url(url: str) -> bool:
    """Return True if the URL is a valid YouTube video URL."""
    try:
        extract_video_id(url)
        return True
    except InvalidYouTubeURLError:
        return False


def build_canonical_watch_url(video_id: str) -> str:
    """Return the canonical YouTube watch URL for a video ID."""
    return f"https://www.youtube.com/watch?v={video_id}"


def build_youtube_timestamp_url(video_id: str, start_seconds: Optional[float]) -> str:
    """Return a YouTube URL pointing to a specific timestamp in seconds."""
    base_url = build_canonical_watch_url(video_id)
    if start_seconds is None:
        return base_url
    seconds_int = max(0, int(start_seconds))
    return f"{base_url}&t={seconds_int}s"


def format_timestamp(seconds: Optional[float]) -> str:
    """Format seconds into MM:SS or H:MM:SS."""
    if seconds is None:
        return "N/A"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fetch_video_metadata(video_id: str, original_url: Optional[str] = None) -> VideoMetadata:
    """Fetch YouTube video title, channel name, and thumbnail URL.

    Uses YouTube's public oEmbed endpoint with an HTML OpenGraph fallback so
    videos with embedding disabled still resolve cleanly.
    """
    watch_url = build_canonical_watch_url(video_id)
    default_thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    title = f"YouTube Video ({video_id})"
    channel = "Unknown Channel"
    thumbnail_url = default_thumbnail

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
    }

    # Primary: YouTube public oEmbed JSON
    try:
        oembed_url = f"https://www.youtube.com/oembed?url={watch_url}&format=json"
        resp = requests.get(oembed_url, headers=headers, timeout=8)
        if resp.status_code == 200:
            payload = resp.json()
            title = payload.get("title") or title
            channel = payload.get("author_name") or channel
            thumbnail_url = payload.get("thumbnail_url") or thumbnail_url
            return VideoMetadata(
                video_id=video_id,
                url=original_url.strip() if original_url else watch_url,
                title=title,
                channel=channel,
                thumbnail_url=thumbnail_url,
            )
    except (requests.RequestException, ValueError):
        pass

    # Fallback: OpenGraph metadata from watch page
    try:
        page_resp = requests.get(watch_url, headers=headers, timeout=8)
        if page_resp.status_code == 200:
            soup = BeautifulSoup(page_resp.text, "html.parser")
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()
            elif soup.title and soup.title.string:
                title = soup.title.string.replace(" - YouTube", "").strip()

            og_image = soup.find("meta", property="og:image")
            if og_image and og_image.get("content"):
                thumbnail_url = og_image["content"].strip()

            author_meta = soup.find("link", itemprop="name")
            if author_meta and author_meta.get("content"):
                channel = author_meta["content"].strip()
    except requests.RequestException:
        pass

    return VideoMetadata(
        video_id=video_id,
        url=original_url.strip() if original_url else watch_url,
        title=title,
        channel=channel,
        thumbnail_url=thumbnail_url,
    )


def count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    if not text:
        return 0
    return len(text.split())


def extract_query_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a user query for passage highlighting."""
    tokens = re.findall(r"[A-Za-z0-9']{3,}", query.lower())
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        cleaned = token.strip("'")
        if len(cleaned) >= 3 and cleaned not in STOPWORDS and cleaned not in seen:
            seen.add(cleaned)
            keywords.append(cleaned)
    return keywords


def highlight_query_terms(text: str, query: str) -> str:
    """Escape text for HTML rendering and wrap matching query terms in <mark>."""
    escaped = html.escape(text)
    keywords = extract_query_keywords(query)
    if not keywords:
        return escaped

    sorted_terms = sorted(keywords, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(term) for term in sorted_terms) + r"[a-z]*)\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub(r'<mark class="rag-highlight">\1</mark>', escaped)
