from pathlib import Path

from app.config import Settings
from app.core.agents.critic_agent import CriticAgent
from app.core.agents.formulator_agent import FormulatorAgent
from app.core.clients.db.rag.vector_store import VectorStore
from app.core.clients.llm import GeminiLLMClient, OllamaLLMClient
from app.core.clients.llm.base import BaseLLMClient
from app.core.service.proof_service import ProofService
from app.core.service.verification_pipeline import VerificationPipeline
from app.core.utils.proof_utils import ProofUtils
from app.services.jobs import InMemoryJobStore
from app.services.proof_application import ProofApplicationService


class ServiceContainer:
    def __init__(self, settings: Settings, project_root: Path) -> None:
        self.settings = settings
        self.project_root = project_root

        self.llm_client: BaseLLMClient = self._build_llm_client()
        self.vector_store = VectorStore()
        self.proof_utils = ProofUtils(self.llm_client)
        self.formulator_agent = FormulatorAgent(
            self.vector_store, self.llm_client, self.proof_utils
        )
        self.critic_agent = CriticAgent(
            self.vector_store, self.llm_client, self.proof_utils
        )
        self.verification_pipeline = VerificationPipeline(
            formulator=self.formulator_agent,
            critic=self.critic_agent,
            proof_utils=self.proof_utils,
        )
        self.proof_service = ProofService(
            pipeline=self.verification_pipeline,
            knowledge_base_dir=project_root
            / "app"
            / "core"
            / "clients"
            / "db"
            / "knowledge_base",
        )
        self.job_store = InMemoryJobStore(ttl_seconds=settings.JOB_STORE_TTL_SECONDS)
        self.proof_application_service = ProofApplicationService(
            self.proof_service, self.job_store
        )

    def _build_llm_client(self) -> BaseLLMClient:
        backend = self.settings.LLM_BACKEND.lower().strip()
        if backend == "ollama":
            return OllamaLLMClient(
                base_url=self.settings.OLLAMA_BASE_URL,
                model=self.settings.OLLAMA_MODEL,
                timeout=self.settings.OLLAMA_TIMEOUT,
            )
        if backend == "gemini":
            return GeminiLLMClient(
                api_key=self.settings.GEMINI_API_KEY,
                model=self.settings.GEMINI_MODEL,
                max_output_tokens=self.settings.LLM_MAX_TOKENS,
                temperature=self.settings.LLM_TEMPERATURE,
            )
        raise ValueError(f"Unsupported LLM_BACKEND: {self.settings.LLM_BACKEND}")

    async def aclose(self) -> None:
        await self.llm_client.aclose()
