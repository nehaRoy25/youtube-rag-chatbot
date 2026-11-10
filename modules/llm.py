"""Groq LLM client for strictly grounded YouTube video question answering."""

from dataclasses import dataclass
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    NotFoundError,
    RateLimitError,
)

import config
from modules.retrieval import RetrievalResult, Retriever

SYSTEM_PROMPT = """You are a precise, helpful AI assistant answering questions strictly about a YouTube video using ONLY its retrieved transcript excerpts.

Mandatory Grounding Rules:
1. Answer ONLY using facts explicitly stated in the "Retrieved Video Transcript Context" provided to you.
2. Do NOT use external knowledge, assumptions, or general background facts not present in the excerpts.
3. Do NOT hallucinate, extrapolate, or invent details.
4. If the retrieved transcript excerpts are in another language (e.g., Hindi, Spanish), understand them accurately and answer in the language of the user's question while staying strictly faithful to the transcript.
5. If the retrieved transcript excerpts do not contain enough information to answer the question, respond clearly with:
   "I couldn't find this information in the YouTube video."
6. Cite your sources inline using the exact chunk IDs and timestamps from the context (for example: `[chunk_003, 02:15]`).
7. Keep answers clear, well-structured, concise, and faithful to what the speaker said."""

_NON_CHAT_MODEL_KEYWORDS = (
    "whisper",
    "orpheus",
    "prompt-guard",
    "safeguard",
    "guard",
    "tts",
    "embedding",
    "moderation",
)

_PREFERRED_MODEL_ORDER = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "allam-2-7b",
]

_MODEL_CACHE: dict[str, object] = {"timestamp": 0.0, "models": []}


class LLMError(Exception):
    """Raised when Groq answer generation fails or configuration is invalid."""


@dataclass
class LLMResponse:
    """Structured response returned by Groq LLM."""

    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _resolve_api_key() -> str:
    """Resolve Groq API key securely from environment / .env without ever exposing it."""
    load_dotenv(dotenv_path=config.BASE_DIR / ".env", override=False)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        raise LLMError(
            "Groq API key is not configured. Please add GROQ_API_KEY to your .env file "
            "and restart or retry."
        )
    return key


def _is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(token in lowered for token in _NON_CHAT_MODEL_KEYWORDS)


def _sort_models(model_ids: list[str]) -> list[str]:
    priority = {mid: idx for idx, mid in enumerate(_PREFERRED_MODEL_ORDER)}
    return sorted(model_ids, key=lambda mid: (priority.get(mid, 999), mid))


def fetch_available_groq_models(force_refresh: bool = False) -> list[str]:
    """Fetch active chat-completion models from Groq API with safe fallback."""
    now = time.time()
    cached_models = _MODEL_CACHE.get("models")
    cached_ts = float(_MODEL_CACHE.get("timestamp", 0.0) or 0.0)
    if not force_refresh and cached_models and (now - cached_ts) < 300:
        return list(cached_models)  # type: ignore[arg-type]

    try:
        api_key = _resolve_api_key()
        client = Groq(api_key=api_key)
        response = client.models.list()
        active_chat_models = [
            m.id
            for m in response.data
            if getattr(m, "active", True) and _is_chat_model(m.id)
        ]
        if active_chat_models:
            ordered = _sort_models(active_chat_models)
            _MODEL_CACHE["timestamp"] = now
            _MODEL_CACHE["models"] = ordered
            return ordered
    except Exception:
        pass

    return list(config.AVAILABLE_GROQ_MODELS)


def _clean_model_output(raw_text: str) -> str:
    """Strip any reasoning <think>...</think> blocks emitted by reasoning models."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


class GroqAnswerGenerator:
    """Generates strictly grounded answers using the Groq Chat Completions API."""

    def __init__(
        self,
        model: str = config.DEFAULT_GROQ_MODEL,
        temperature: float = config.DEFAULT_TEMPERATURE,
        max_tokens: int = config.DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call_groq(self, client: Groq, model_name: str, messages: list[dict[str, str]]):
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def generate_answer(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        video_title: str = "",
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> LLMResponse:
        """Send retrieved transcript context and user question to Groq LLM."""
        if not question or not question.strip():
            raise LLMError("Question cannot be empty.")

        if not retrieval_result.results:
            return LLMResponse(
                answer="I couldn't find this information in the YouTube video.",
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

        api_key = _resolve_api_key()
        client = Groq(api_key=api_key)

        formatted_context = Retriever.format_context_for_llm(retrieval_result)
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if chat_history:
            for turn in chat_history[-6:]:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})

        title_line = f"Video Title: {video_title}\n" if video_title else ""
        user_prompt = (
            f"{title_line}"
            f"Retrieved Video Transcript Context:\n"
            f"===================================\n"
            f"{formatted_context}\n"
            f"===================================\n\n"
            f"User Question: {question.strip()}\n\n"
            f"Instructions: Answer using ONLY the transcript context above. "
            f"Cite chunk IDs and timestamps (e.g., [chunk_001, 00:45]). "
            f'If the answer is not in the context, reply: "I couldn\'t find this information in the YouTube video."'
        )
        messages.append({"role": "user", "content": user_prompt})

        active_model = self.model
        try:
            completion = self._call_groq(client, active_model, messages)
        except (NotFoundError, APIStatusError) as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 404 or isinstance(exc, NotFoundError):
                available = fetch_available_groq_models(force_refresh=True)
                fallback_model = next((m for m in available if m != active_model), None)
                if fallback_model:
                    try:
                        active_model = fallback_model
                        completion = self._call_groq(client, active_model, messages)
                    except Exception as retry_exc:
                        raise LLMError(
                            f"Selected model '{self.model}' was unavailable (404) and fallback "
                            f"'{fallback_model}' failed: {retry_exc}"
                        ) from retry_exc
                else:
                    raise LLMError(
                        f"Groq model '{self.model}' was not found (404). "
                        f"Available models: {', '.join(available)}"
                    ) from exc
            else:
                raise LLMError(
                    f"Groq API returned status {status_code}. Please verify the selected model '{active_model}'."
                ) from exc
        except AuthenticationError as exc:
            raise LLMError(
                "Groq authentication failed. Please verify GROQ_API_KEY in your .env file."
            ) from exc
        except RateLimitError as exc:
            raise LLMError(
                "Groq API rate limit reached. Please wait a moment and try again."
            ) from exc
        except APIConnectionError as exc:
            raise LLMError(
                "Could not connect to the Groq API. Check your internet connection."
            ) from exc
        except Exception as exc:
            raise LLMError(f"Unexpected error generating answer with Groq: {exc}") from exc

        choice = completion.choices[0] if completion.choices else None
        raw_content = (
            choice.message.content
            if choice and choice.message and choice.message.content
            else ""
        )
        answer_text = _clean_model_output(raw_content)
        if not answer_text:
            answer_text = "I couldn't find this information in the YouTube video."

        usage = getattr(completion, "usage", None)
        return LLMResponse(
            answer=answer_text,
            model=active_model,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )
