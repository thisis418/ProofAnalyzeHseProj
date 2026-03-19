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


class UsedTheoremSchema(BaseModel):
    fact_name: str | None = None
    theorem_name: str
    matched_theorem: str | None = None
    references: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class ProofStepSchema(BaseModel):
    content: str = ""
    content_latex: str = ""
    justification: str = ""
    step_type: str = "assertion"
    source_indices: list[int] = Field(default_factory=list)
    line_number: int | None = None
    used_theorems: list[UsedTheoremSchema] = Field(default_factory=list)


class VerifyProofResponseSchema(BaseModel):
    proof_id: str
    is_valid: bool
    confidence_score: float
    summary: str
    iteration_recommendation: str | None = None
    parsed_steps: list[ProofStepSchema]
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
