r"""Async Gemini client with safe JSON parsing and retry logic."""

import asyncio
import collections
import json
import logging
import re
import threading
import time
from typing import Any, Deque

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - optional import during local development
    genai = None
    genai_errors = None
    genai_types = None

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: Deque[float] = collections.deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                cutoff = now - self._window
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return
                oldest = self._timestamps[0]
                wait = (oldest + self._window) - now + 0.1
                logger.info(
                    "[rate limiter] %s/%s requests in window. Sleeping %.1fs...",
                    len(self._timestamps),
                    self._max,
                    wait,
                )
                self._lock.release()
                time.sleep(wait)
                self._lock.acquire()


class GeminiLLMClient:
    DEFAULT_RATE_LIMIT_RPM: int = 12
    RATE_WINDOW_SECONDS: float = 60.0
    RETRY_MAX_ATTEMPTS: int = 6
    RETRY_BASE_DELAY: float = 16.0
    RETRY_MAX_DELAY: float = 120.0

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-1.5-flash",
        max_output_tokens: int = 4000,
        temperature: float = 0.1,
        rate_limit_rpm: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.rate_limit_rpm = rate_limit_rpm or self.DEFAULT_RATE_LIMIT_RPM
        self._client = (
            genai.Client(api_key=self.api_key) if genai and self.api_key else None
        )
        self._rate_limiter = SlidingWindowRateLimiter(
            self.rate_limit_rpm, self.RATE_WINDOW_SECONDS
        )

    @staticmethod
    def _extract_json_candidate(raw: str) -> str:
        raw = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        return (
            raw[start : end + 1] if start != -1 and end != -1 and end > start else raw
        )

    @staticmethod
    def _fix_json_escapes(raw: str) -> str:
        result: list[str] = []
        in_string = False
        escaped = False
        index = 0
        while index < len(raw):
            char = raw[index]
            if not in_string:
                result.append(char)
                if char == '"':
                    in_string = True
                index += 1
                continue
            if escaped:
                if char in {'"', "\\", "/", "b", "f", "n", "r", "t"} or char == "u":
                    result.append(char)
                else:
                    result.append("\\")
                    result.append(char)
                escaped = False
                index += 1
                continue
            if char == "\\":
                result.append(char)
                escaped = True
                index += 1
                continue
            result.append(char)
            if char == '"':
                in_string = False
            index += 1
        if escaped:
            result.append("\\")
        return "".join(result)

    @classmethod
    def _parse_json_safe(cls, raw: str) -> dict[str, Any]:
        candidates = [raw, cls._extract_json_candidate(raw)]
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        repaired = [
            cls._fix_json_escapes(raw),
            cls._fix_json_escapes(cls._extract_json_candidate(raw)),
        ]
        for candidate in repaired:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        logger.error(
            "JSON parse failed after repair. Raw(first 1000)=%r Repaired(first 1000)=%r",
            raw[:1000],
            repaired[-1][:1000],
        )
        return {
            "error": "json_decode_failed",
            "raw_preview": raw[:1000],
            "repaired_preview": repaired[-1][:1000],
        }

    @staticmethod
    def _extract_retry_delay(error: Exception) -> float:
        try:
            details = error.args[0] if getattr(error, "args", None) else {}
            if isinstance(details, dict):
                for item in details.get("error", {}).get("details", []):
                    delay_str = item.get("retryDelay", "")
                    if delay_str:
                        seconds = float(re.sub(r"[^0-9.]", "", delay_str))
                        return min(seconds + 2.0, GeminiLLMClient.RETRY_MAX_DELAY)
        except Exception:
            pass
        return GeminiLLMClient.RETRY_BASE_DELAY

    def _call_sync(self, prompt: str, system_instruction: str = "") -> dict[str, Any]:
        if self._client is None or genai_types is None:
            logger.warning("Gemini client not initialised. Returning mock response.")
            return {"mock": True}
        config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            system_instruction=system_instruction or None,
        )
        last_error: Exception | None = None
        for attempt in range(1, self.RETRY_MAX_ATTEMPTS + 1):
            self._rate_limiter.acquire()
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                raw = response.text
                if not raw:
                    return {"error": "empty_response"}
                return self._parse_json_safe(raw)
            except Exception as exc:  # pragma: no cover - network/provider dependent
                last_error = exc
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if status == 429:
                    delay = self._extract_retry_delay(exc)
                    if attempt < self.RETRY_MAX_ATTEMPTS:
                        time.sleep(delay)
                        continue
                    return {"error": f"rate_limit_exhausted: {exc}"}
                if status in (500, 503):
                    delay = min(self.RETRY_BASE_DELAY * attempt, self.RETRY_MAX_DELAY)
                    if attempt < self.RETRY_MAX_ATTEMPTS:
                        time.sleep(delay)
                        continue
                logger.exception("Unexpected error during Gemini LLM call")
                return {"error": str(exc)}
        return {"error": f"all_retries_failed: {last_error}"}

    async def call(self, prompt: str, system_instruction: str = "") -> dict[str, Any]:
        return await asyncio.to_thread(self._call_sync, prompt, system_instruction)

    async def aclose(self) -> None:
        return None
