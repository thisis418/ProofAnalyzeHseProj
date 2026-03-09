import asyncio
from uuid import UUID

from app.api.schemas import VerifyProofRequestSchema
from app.core.service.proof_service import ProofService
from app.services.jobs import InMemoryJobStore, JobRecord


class ProofApplicationService:
    def __init__(
        self, proof_service: ProofService, job_store: InMemoryJobStore
    ) -> None:
        self._proof_service = proof_service
        self._job_store = job_store

    async def verify_proof(self, payload: VerifyProofRequestSchema) -> dict:
        return await self._proof_service.analyze_proof(
            payload.proof_id,
            payload.latex,
            payload.context.model_dump(),
            payload.max_rounds,
        )

    async def get_theorems(self, names: list[str]) -> list[dict]:
        return await self._proof_service.get_theorems(names)

    async def create_job(self, payload: VerifyProofRequestSchema) -> JobRecord:
        job = await self._job_store.create(payload.model_dump())
        asyncio.create_task(self._run_job(job.job_id, payload))
        return job

    async def get_job(self, job_id: UUID) -> JobRecord | None:
        return await self._job_store.get(job_id)

    async def _run_job(self, job_id: UUID, payload: VerifyProofRequestSchema) -> None:
        await self._job_store.mark_running(job_id)
        try:
            result = await self.verify_proof(payload)
            await self._job_store.mark_completed(job_id, result)
        except Exception as exc:  # pragma: no cover - async background task
            await self._job_store.mark_failed(job_id, str(exc))
