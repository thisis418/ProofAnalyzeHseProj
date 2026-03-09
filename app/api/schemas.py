from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ProofContextSchema(BaseModel):
    topic: str = Field(default="математика")
    level: str = Field(default="undergraduate")


class VerifyProofRequestSchema(BaseModel):
    proof_id: str = Field(min_length=1)
    latex: str = Field(min_length=1)
    context: ProofContextSchema = Field(default_factory=ProofContextSchema)
    max_rounds: int | None = Field(default=None, ge=1, le=10)


class VerifyProofResponseSchema(BaseModel):
    proof_id: str
    is_valid: bool
    confidence_score: float
    summary: str
    iteration_recommendation: str | None = None
    parsed_steps: list[dict[str, Any]]
    phases: dict[str, Any]
    remarks: list[dict[str, Any]]
    debate_history: list[dict[str, Any]]


class CreateJobResponseSchema(BaseModel):
    job_id: UUID
    status: Literal["queued", "running", "completed", "failed"]


class JobStatusResponseSchema(BaseModel):
    job_id: UUID
    status: Literal["queued", "running", "completed", "failed"]
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    result: dict[str, Any] | None = None


class TheoremResponseSchema(BaseModel):
    name: str
    type: str | None = None
    statement: str | None = None
    latex: str | None = None
    category: str | None = None
