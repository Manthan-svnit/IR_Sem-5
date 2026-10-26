"""
Thin wrapper around the `google-genai` SDK (the successor to the now-deprecated
`google-generativeai` package). Two responsibilities:

  • embed_documents / embed_query  — gemini-embedding-001 (768-d, L2-normalized)
  • generate_grounded_answer       — gemini-2.5-flash chat with RAG context
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
import time
from functools import lru_cache
from typing import Callable, Iterable, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

logger = logging.getLogger("chatpdf.gemini")

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# --- Chat Models (Uncomment whichever model you want to set as primary) ---
CHAT_MODEL  = os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-lite-latest")  # Currently active: low latency & reliable capacity
# CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.1-flash-lite")     # High speed lite model (may experience traffic spikes)
# CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")          # Powerful flash model (20 req/day free limit)
# CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")          # Legacy 2.5 flash
# CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")          # Legacy 2.0 flash

# Automatic failover order if active model returns 404, 429 (rate limit), or 503 (high demand)
FALLBACK_CHAT_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
]
# gemini-embedding-001 supports Matryoshka — request 768 to match our pgvector schema.
EMBED_DIM   = 768

_client: genai.Client | None = None


T = TypeVar("T")

_RETRY_AFTER_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


def _retry_delay_from_error(err: Exception) -> float | None:
    """Pull the server-suggested retry delay out of a 429 message, if present."""
    m = _RETRY_AFTER_RE.search(str(err))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _with_retry(fn: Callable[[], T], op: str, max_attempts: int = 5) -> T:
    """
    Run `fn` with exponential backoff on 429 / 5xx.
    Honours the `retry in Ns` hint Gemini includes in 429 responses.
    """
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except genai_errors.ClientError as e:
            # 429 RESOURCE_EXHAUSTED — wait and retry. The SDK exposes the
            # HTTP status as `.code` (int); `.status` is the proto status name.
            code = getattr(e, "code", None)
            if code == 429 and attempt < max_attempts:
                hint = _retry_delay_from_error(e)
                # Server hint, capped at 30s; otherwise exponential backoff with jitter.
                wait = min(hint, 30) if hint else min(2 ** attempt + random.random(), 30)
                logger.warning(
                    f"[{op}] 429 from Gemini (attempt {attempt}/{max_attempts}); "
                    f"sleeping {wait:.1f}s"
                )
                time.sleep(wait)
                last_err = e
                continue
            raise
        except genai_errors.ServerError as e:
            if attempt < max_attempts:
                wait = min(2 ** attempt + random.random(), 30)
                logger.warning(
                    f"[{op}] {e} (attempt {attempt}/{max_attempts}); sleeping {wait:.1f}s"
                )
                time.sleep(wait)
                last_err = e
                continue
            raise
    assert last_err is not None
    raise last_err


def _l2_normalize(vec: list[float]) -> list[float]:
    """gemini-embedding-001 at <3072 dims requires L2 normalization for cosine similarity."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _get_client() -> genai.Client:
    """Lazily build the SDK client. Accepts GEMINI_API_KEY or GOOGLE_API_KEY."""
    global _client
    if _client is None:
        key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        ).strip().strip('"').strip("'")
        if not key or key in ("YOUR_GEMINI_KEY", "YOUR_GEMINI_API_KEY"):
            raise EnvironmentError(
                "GEMINI_API_KEY is not configured or is a placeholder in backend/.env. "
                "Please get a free API key from https://aistudio.google.com/app/apikey"
            )
        _client = genai.Client(api_key=key)
    return _client


def _call_chat_model(
    client: genai.Client,
    contents: Any,
    config: types.GenerateContentConfig | None = None,
    op: str = "chat",
):
    """Call the chat model with fallback if the requested model is unavailable (404), quota exhausted (429), or overloaded (503)."""
    global CHAT_MODEL
    candidates = [CHAT_MODEL] + [m for m in FALLBACK_CHAT_MODELS if m != CHAT_MODEL]
    last_exc = None
    for m in candidates:
        try:
            res = client.models.generate_content(
                model=m,
                contents=contents,
                config=config,
            )
            if m != CHAT_MODEL:
                logger.info(f"[{op}] Switched active model from '{CHAT_MODEL}' to '{m}'")
                CHAT_MODEL = m
            return res
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            code = getattr(e, "code", None)
            if code in (404, 429, 503):
                logger.warning(
                    f"[{op}] Model '{m}' returned {code} ({getattr(e, 'status', 'ERR')}), trying next candidate..."
                )
                last_exc = e
                continue
            raise
    if last_exc:
        raise last_exc


def verify_gemini_key() -> tuple[bool, str]:
    """Check if the Gemini API key is configured and can authenticate."""
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip().strip('"').strip("'")
    if not key or key in ("YOUR_GEMINI_KEY", "YOUR_GEMINI_API_KEY"):
        return False, "GEMINI_API_KEY is not configured or set to placeholder in .env"
    try:
        client = _get_client()
        _call_chat_model(client, contents="ping", op="verify_key")
        return True, f"Key verified with model '{CHAT_MODEL}'."
    except Exception as e:
        return False, str(e)


def embed_documents(texts: Iterable[str]) -> list[list[float]]:
    """
    Embed a batch of passages for storage/retrieval.
    Uses task_type=RETRIEVAL_DOCUMENT which Gemini optimises for indexing.
    """
    texts = list(texts)
    if not texts:
        return []

    client = _get_client()
    out: list[list[float]] = []
    # gemini-embedding-001 free tier is ~5–100 RPM; 8 per call keeps token volume
    # low enough that TPM limits don't trip on long chunks.
    BATCH = 8
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]

        def _call(b=batch):
            return client.models.embed_content(
                model=EMBED_MODEL,
                contents=b,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBED_DIM,
                ),
            )

        resp = _with_retry(_call, op=f"embed_documents[{i}:{i+len(batch)}]")
        for emb in resp.embeddings:
            out.append(_l2_normalize(list(emb.values)))
    return out


@lru_cache(maxsize=128)
def _cached_embed_query_tuple(text: str) -> tuple[float, ...]:
    client = _get_client()
    resp = _with_retry(
        lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBED_DIM,
            ),
        ),
        op="embed_query",
    )
    return tuple(_l2_normalize(list(resp.embeddings[0].values)))


def embed_query(text: str) -> list[float]:
    """Embed a single user query with task_type=RETRIEVAL_QUERY (cached)."""
    return list(_cached_embed_query_tuple(text.strip()))


_REFERENCE_PATTERN = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|theirs|he|him|his|she|her|hers|"
    r"former|latter|above|previous|earlier|again|same|also|too|more|else)\b",
    re.IGNORECASE,
)


def should_condense(question: str, history: list[dict] | None) -> bool:
    """
    Fast check: only call LLM condensation if the question is genuinely dependent on history.
    Saves an entire LLM inference round-trip (~2-4s) on standalone questions.
    """
    if not history:
        return False
    words = question.strip().split()
    # Very short follow-ups (e.g. "Why?", "What about profits?") need context
    if len(words) <= 4:
        return True
    # If the user uses referential pronouns, condense it
    if _REFERENCE_PATTERN.search(question):
        return True
    return False


CONDENSE_INSTRUCTION = (
    "You rewrite a follow-up question into a standalone question that can be "
    "understood without the prior conversation. Resolve any pronoun or "
    'demonstrative reference ("it", "that", "this", "those", "those changes") '
    "using the conversation. Preserve every concrete noun and qualifier from "
    "the follow-up. If the follow-up is already self-contained, return it "
    "unchanged. Output exactly one line containing only the rewritten "
    "question. Do not add quotes, preface, explanation, or trailing punctuation "
    "beyond a single question mark."
)


CONDENSE_FEWSHOT = (
    "Example:\n"
    "Conversation so far:\n"
    "User: Summarize airport performance in H1-26.\n"
    "Assistant: Passenger traffic rose to 49.3M while cargo reached 0.62M MT.\n"
    "Follow-up: Break that down into passenger and cargo changes.\n"
    "Standalone: Break down airport performance in H1-26 into passenger and cargo changes.\n\n"
)


def condense_question(question: str, history: list[dict]) -> str:
    """
    Rewrite a follow-up into a standalone question so retrieval gets useful
    context. Skips LLM call if question is already standalone.
    """
    if not should_condense(question, history):
        return question

    client = _get_client()

    # Build a compact transcript using only the last four turns
    transcript_lines: list[str] = []
    for msg in history[-4:]:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        transcript_lines.append(f"{speaker}: {msg['content']}")
    transcript = "\n".join(transcript_lines)

    prompt = (
        f"{CONDENSE_FEWSHOT}"
        "Now rewrite this:\n"
        f"Conversation so far:\n{transcript}\n"
        f"Follow-up: {question}\n"
        "Standalone:"
    )

    try:
        resp = _with_retry(
            lambda: _call_chat_model(
                client=client,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=CONDENSE_INSTRUCTION,
                    temperature=0.0,
                    max_output_tokens=128,
                ),
                op="condense_question",
            ),
            op="condense_question",
        )
        condensed = (resp.text or "").strip().strip('"').strip("'").rstrip()
        # Strip a leading "Standalone:" or "Question:" if the model echoed it.
        for prefix in ("Standalone:", "Question:", "Q:"):
            if condensed.lower().startswith(prefix.lower()):
                condensed = condensed[len(prefix):].strip()
        if not condensed:
            return question
        logger.info(f"[CONDENSE] {question!r} -> {condensed!r}")
        return condensed
    except Exception as e:
        logger.warning(f"[CONDENSE] failed ({e}); using raw question")
        return question


SYSTEM_INSTRUCTION = (
    "You are a precise document assistant. Answer the user's question using ONLY "
    "the numbered context blocks provided. After every sentence that uses a fact "
    "from a block, append its citation in square brackets like [1] or [2,5]. "
    "If the answer is not present in the context, reply exactly: "
    '"I could not find that in the document." '
    "Do not invent citations. Do not reference your own knowledge. "
    "Write clearly and concisely."
)


def _build_context_block(retrieved: list[dict]) -> str:
    parts = []
    for i, r in enumerate(retrieved, start=1):
        snippet = r["text"].strip().replace("\n\n", "\n")
        parts.append(f"[{i}] (page {r['page']})\n{snippet}")
    return "\n\n".join(parts)


def _history_to_contents(history: list[dict]) -> list[types.Content]:
    """Map our {role: user/assistant, content: str} history to genai Contents."""
    contents: list[types.Content] = []
    for msg in history[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )
    return contents


def generate_grounded_answer(
    question: str,
    retrieved: list[dict],
    history: list[dict] | None = None,
) -> str:
    """
    Call Gemini with the retrieved context + last few turns of chat history.

    `retrieved` is the list returned by vector_store.match_chunks.
    `history` is [{role, content}, …] — only the last six turns are sent.
    """
    history = history or []
    client = _get_client()

    contents = _history_to_contents(history)
    context_block = _build_context_block(retrieved)
    final_user_msg = (
        f"Context blocks from the document:\n\n{context_block}\n\n"
        f"Question: {question}"
    )
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=final_user_msg)])
    )

    resp = _with_retry(
        lambda: _call_chat_model(
            client=client,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,
                max_output_tokens=768,
            ),
            op="generate_grounded_answer",
        ),
        op="generate_grounded_answer",
    )
    return (resp.text or "").strip()
