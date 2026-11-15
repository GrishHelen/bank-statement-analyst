"""Выбор провайдера модели по настройке LLM_PROVIDER, без правок кода."""

from typing import Any, Protocol

from src.chat import Message, Reply
from src.config import Settings


class ChatModel(Protocol):
    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Reply: ...

    def __enter__(self) -> "ChatModel": ...

    def __exit__(self, *args: Any) -> None: ...


def create_model(settings: Settings) -> ChatModel:
    if settings.provider == "openai":
        from src.providers.openai_compatible import OpenAICompatibleModel

        return OpenAICompatibleModel(settings)
    from src.providers.gigachat import GigaChatModel

    return GigaChatModel(settings)


def model_label(settings: Settings) -> str:
    if settings.provider == "openai":
        return f"{settings.openai_model} (OpenAI-совместимый API)"
    return f"{settings.model} (GigaChat)"
