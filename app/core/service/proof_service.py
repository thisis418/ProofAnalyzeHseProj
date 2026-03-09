import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from app.core.clients.db.rag.loader import load_knowledge_from_path
from app.core.service.verification_pipeline import VerificationPipeline


class ProofService:
    def __init__(
        self, pipeline: VerificationPipeline, knowledge_base_dir: Path
    ) -> None:
        self._pipeline = pipeline
        self._knowledge_base_dir = knowledge_base_dir

    async def analyze_proof(
        self,
        proof_id: str,
        latex: str,
        context: dict[str, Any] | None = None,
        max_rounds: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if max_rounds is not None:
            kwargs["max_rounds_per_phase"] = max_rounds
        return await self._pipeline.verify_proof(
            proof_id=proof_id, latex=latex, context=context or {}, **kwargs
        )

    @lru_cache(maxsize=1)
    def _load_knowledge_sync(self) -> list[dict[str, Any]]:
        return load_knowledge_from_path(self._knowledge_base_dir)

    async def get_theorems(self, names: Sequence[str]) -> list[dict[str, Any]]:
        requested = {name.strip() for name in names if name.strip()}
        items = await asyncio.to_thread(self._load_knowledge_sync)
        return [
            {
                "name": item.get("name"),
                "type": item.get("type"),
                "statement": item.get("statement"),
                "latex": item.get("latex"),
                "category": item.get("category"),
            }
            for item in items
            if item.get("name") in requested
        ]
