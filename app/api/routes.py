from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_proof_application_service
from app.api.schemas import (
    CreateJobResponseSchema,
    JobStatusResponseSchema,
    TheoremResponseSchema,
    VerifyProofRequestSchema,
    VerifyProofResponseSchema,
)
from app.services.proof_application import ProofApplicationService

router = APIRouter()


@router.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/proofs/verify", response_model=VerifyProofResponseSchema)
async def verify_proof(
    payload: VerifyProofRequestSchema,
    service: ProofApplicationService = Depends(get_proof_application_service),
) -> VerifyProofResponseSchema:
    result = await service.verify_proof(payload)
    if "error" in result and result.get("summary") is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["error"]
        )
    return VerifyProofResponseSchema(**result)


@router.post(
    "/proofs/jobs",
    response_model=CreateJobResponseSchema,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_verification_job(
    payload: VerifyProofRequestSchema,
    service: ProofApplicationService = Depends(get_proof_application_service),
) -> CreateJobResponseSchema:
    job = await service.create_job(payload)
    return CreateJobResponseSchema(job_id=job.job_id, status=job.status)


@router.get("/proofs/jobs/{job_id}", response_model=JobStatusResponseSchema)
async def get_verification_job(
    job_id: str,
    service: ProofApplicationService = Depends(get_proof_application_service),
) -> JobStatusResponseSchema:
    from uuid import UUID

    job = await service.get_job(UUID(job_id))
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )
    return JobStatusResponseSchema(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
        result=job.result,
    )


@router.get("/theorems", response_model=list[TheoremResponseSchema])
async def get_theorems(
    names: str = Query(default=""),
    service: ProofApplicationService = Depends(get_proof_application_service),
) -> list[TheoremResponseSchema]:
    name_list = [item for item in names.split("|") if item]
    result = await service.get_theorems(name_list)
    return [TheoremResponseSchema(**item) for item in result]
