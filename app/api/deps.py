from fastapi import Depends, Request

from app.core.containers.container import ServiceContainer
from app.services.proof_application import ProofApplicationService


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def get_proof_application_service(
    container: ServiceContainer = Depends(get_container),
) -> ProofApplicationService:
    return container.proof_application_service
