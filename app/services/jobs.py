import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import Any, Literal
from uuid import UUID, uuid4


JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class JobRecord:
    job_id: UUID
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


class InMemoryJobStore:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = ttl_seconds
        self._jobs: dict[UUID, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, payload: dict[str, Any]) -> JobRecord:
        async with self._lock:
            now = datetime.now(UTC)
            job = JobRecord(
                job_id=uuid4(),
                status="queued",
                created_at=now,
                updated_at=now,
                payload=payload,
            )
            self._jobs[job.job_id] = job
            self._cleanup_locked(now)
            return job

    async def get(self, job_id: UUID) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def mark_running(self, job_id: UUID) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.updated_at = datetime.now(UTC)

    async def mark_completed(self, job_id: UUID, result: dict[str, Any]) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.result = result
            job.updated_at = datetime.now(UTC)

    async def mark_failed(self, job_id: UUID, error: str) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.error = error
            job.updated_at = datetime.now(UTC)

    def _cleanup_locked(self, now: datetime) -> None:
        threshold = now - timedelta(seconds=self._ttl)
        expired = [
            job_id for job_id, job in self._jobs.items() if job.updated_at < threshold
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)
