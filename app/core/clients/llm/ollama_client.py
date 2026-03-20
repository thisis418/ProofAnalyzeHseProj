import json
import logging
import re
import asyncio
from typing import Any

import httpx

from app.core.clients.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OllamaLLMClient(BaseLLMClient):
    RETRY_ATTEMPTS = 3
    RETRY_BASE_DELAY_SECONDS = 2.0

    def __init__(self, base_url: str, model: str, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    @staticmethod
    def _extract_json_candidate(raw: str) -> str:
        raw = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw[start : end + 1]
        return raw

    @staticmethod
    def _fix_json_escapes(raw: str) -> str:
        result: list[str] = []
        in_string = False
        escaped = False
        i = 0
        while i < len(raw):
            ch = raw[i]
            if not in_string:
                result.append(ch)
                if ch == '"':
                    in_string = True
                i += 1
                continue
            if escaped:
                if ch in {'"', "\\", "/", "b", "f", "n", "r", "t"} or ch == "u":
                    result.append(ch)
                else:
                    result.append("\\")
                    result.append(ch)
                escaped = False
                i += 1
                continue
            if ch == "\\":
                result.append(ch)
                escaped = True
                i += 1
                continue
            result.append(ch)
            if ch == '"':
                in_string = False
            i += 1
        if escaped:
            result.append("\\")
        return "".join(result)

    @classmethod
    def _parse_json_safe(cls, raw: str) -> dict[str, Any]:
        for candidate in (raw, cls._extract_json_candidate(raw)):
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
            "Ollama JSON parse failed. Raw(first 1000)=%r repaired(first 1000)=%r",
            raw[:1000],
            repaired[-1][:1000],
        )
        return {
            "error": "json_decode_failed",
            "raw_preview": raw[:1000],
            "repaired_preview": repaired[-1][:1000],
        }

    async def call(self, prompt: str, system_instruction: str = "") -> dict[str, Any]:
        full_prompt = (
            prompt if not system_instruction else f"{system_instruction}\n\n{prompt}"
        )
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
        }
        last_error: str | None = None
        for attempt in range(1, self.RETRY_ATTEMPTS + 1):
            try:
                response = await self._client.post("/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                raw = data.get("response", "")
                if not raw:
                    return {"error": "empty_response"}
                return self._parse_json_safe(raw)
            except httpx.ReadTimeout:
                last_error = "ollama_timeout"
                logger.warning(
                    "Ollama request timeout (model=%s, attempt=%s/%s)",
                    self.model,
                    attempt,
                    self.RETRY_ATTEMPTS,
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                body_preview = (
                    exc.response.text[:300] if exc.response is not None else ""
                )
                logger.error(
                    "Ollama HTTP error: status=%s body(first 300)=%r",
                    status,
                    body_preview,
                )
                return {"error": f"ollama_http_error:{status}"}
            except httpx.RequestError as exc:
                last_error = "ollama_network_error"
                logger.warning(
                    "Ollama network error (attempt=%s/%s): %s",
                    attempt,
                    self.RETRY_ATTEMPTS,
                    exc,
                )
            except Exception as exc:
                logger.exception("Unexpected Ollama client error")
                return {"error": f"ollama_unexpected_error:{exc}"}

            if attempt < self.RETRY_ATTEMPTS:
                delay = self.RETRY_BASE_DELAY_SECONDS * attempt
                await asyncio.sleep(delay)

        return {"error": last_error or "ollama_call_failed"}

    async def aclose(self) -> None:
        await self._client.aclose()
