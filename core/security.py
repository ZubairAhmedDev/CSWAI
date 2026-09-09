from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass

MAX_QUERY_CHARS = 1200
MAX_CONTEXT_CHARS_PER_CHUNK = 4500
MAX_CHUNKS = 5
RATE_LIMIT_REQUESTS = 12
RATE_LIMIT_WINDOW_SECONDS = 60

SUSPICIOUS_QUERY_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above) instructions",
    r"disregard (all|any|the)?\s*(previous|prior|above) instructions",
    r"forget (all|any|the)?\s*(previous|prior|above) instructions",
    r"reveal (the )?(system|developer) prompt",
    r"show (me )?(the )?(system|developer) prompt",
    r"print (the )?(system|developer) prompt",
    r"what (are|were) your (system|developer) instructions",
    r"act as (the )?(system|developer|administrator|root)",
    r"you are now (the )?(system|developer|administrator|root)",
    r"jailbreak",
    r"bypass (the )?(rules|guardrails|policy|security)",
    r"override (the )?(rules|instructions|policy|security)",
    r"execute (a )?(shell|terminal|cmd|powershell|command)",
    r"read (local|system|private|secret) files",
    r"list (the )?(environment variables|secrets|api keys|tokens)",
    r"exfiltrate",
]

SUSPICIOUS_DOCUMENT_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above) instructions",
    r"system prompt",
    r"developer message",
    r"you are chatgpt",
    r"assistant must",
    r"execute command",
    r"run this command",
    r"reveal secret",
    r"api[_ -]?key",
    r"auth[_ -]?token",
    r"password",
]

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class SecurityDecision:
    allowed: bool
    reason: str = ""
    cleaned_text: str = ""


def normalize_text(text: str) -> str:
    text = CONTROL_CHARS_RE.sub("", text or "")
    text = text.replace("\u202e", "")
    return re.sub(r"\s+", " ", text).strip()


def inspect_user_query(query: str) -> SecurityDecision:
    cleaned = normalize_text(query)

    if not cleaned:
        return SecurityDecision(False, "empty_query", "")

    if len(cleaned) > MAX_QUERY_CHARS:
        return SecurityDecision(
            False,
            f"query_too_long:{len(cleaned)}",
            cleaned[:MAX_QUERY_CHARS],
        )

    lowered = cleaned.lower()

    for pattern in SUSPICIOUS_QUERY_PATTERNS:
        if re.search(pattern, lowered, flags=re.I):
            return SecurityDecision(False, "prompt_injection_pattern", cleaned)

    return SecurityDecision(True, "", cleaned)


def document_chunk_is_suspicious(text: str) -> bool:
    cleaned = normalize_text(text).lower()
    return any(
        re.search(pattern, cleaned, flags=re.I)
        for pattern in SUSPICIOUS_DOCUMENT_PATTERNS
    )


def sanitize_retrieved_chunks(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    safe = []
    blocked = []

    for chunk in (chunks or [])[:MAX_CHUNKS]:
        text = normalize_text(str(chunk.get("text", "")))

        if not text:
            continue

        item = dict(chunk)
        item["text"] = text[:MAX_CONTEXT_CHARS_PER_CHUNK]

        if document_chunk_is_suspicious(item["text"]):
            item["security_block_reason"] = "possible_document_prompt_injection"
            blocked.append(item)
        else:
            safe.append(item)

    return safe, blocked


class SessionRateLimiter:
    def __init__(
        self,
        max_requests: int = RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps = deque()

    def allow(self) -> bool:
        now = time.time()

        while self.timestamps and now - self.timestamps[0] > self.window_seconds:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.max_requests:
            return False

        self.timestamps.append(now)
        return True
