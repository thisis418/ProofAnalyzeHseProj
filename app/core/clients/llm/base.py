from typing import Any, Protocol


class BaseLLMClient(Protocol):
    async def call(
        self, prompt: str, system_instruction: str = ""
    ) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...
