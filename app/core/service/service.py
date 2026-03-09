"""Backward-compatible thin wrappers around the object-oriented services."""

from typing import Any, Sequence

from app.core.service.proof_service import ProofService


def analyze_proof(
    proof_service: ProofService,
    proof_id: str,
    latex: str,
    context: dict[str, Any] | None = None,
    max_rounds: int | None = None,
) -> dict[str, Any]:
    return proof_service.analyze_proof(
        proof_id=proof_id, latex=latex, context=context or {}, max_rounds=max_rounds
    )


def get_theorems(
    proof_service: ProofService, names: Sequence[str]
) -> list[dict[str, Any]]:
    return proof_service.get_theorems(names)
