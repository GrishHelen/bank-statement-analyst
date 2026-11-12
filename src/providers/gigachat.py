"""Подключение к GigaChat и преобразование ответа SDK в текст."""

import ssl
from types import TracebackType

import certifi
import httpx
from gigachat import GigaChat
from gigachat.exceptions import GigaChatException, ResponseError
from gigachat.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatModelOptions,
    ChatToolConfig,
)

from src.chat import ChatError, Message, Reply
from src.config import Settings


API_ERRORS: dict[int, str] = {
    400: "Сервис отклонил запрос. Сократите сообщение или очистите историю командой /reset.",
    401: "Ошибка авторизации. Проверьте GIGACHAT_CREDENTIALS и GIGACHAT_SCOPE.",
    403: "Доступ запрещён. Проверьте права проекта, доступ к модели и остаток квоты.",
    404: "Модель или адрес API не найдены. Проверьте GIGACHAT_MODEL и GIGACHAT_BASE_URL.",
    413: "Слишком большой запрос. Сократите сообщение или используйте /reset.",
    422: "Сервис не принял параметры запроса. Проверьте настройки модели.",
    429: "Достигнут лимит запросов или квота GigaChat. Проверьте кабинет и повторите позже.",
}


class GigaChatModel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            context = ssl.create_default_context()
            context.load_verify_locations(certifi.where())
            if settings.ca_bundle_file:
                context.load_verify_locations(str(settings.ca_bundle_file))
            self.client = GigaChat(
                credentials=settings.credentials,
                scope=settings.scope,
                model=settings.model,
                base_url=settings.base_url,
                auth_url=settings.auth_url,
                timeout=settings.timeout,
                ssl_context=context,
                verify_ssl_certs=True,
                max_retries=0,
            )
        except (OSError, ValueError) as exc:
            raise ChatError("Не удалось настроить GigaChat. Проверьте настройки и PEM-сертификат.") from exc

    def __enter__(self) -> "GigaChatModel":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.client.close()

    def complete(self, messages: list[Message]) -> Reply:
        request = ChatCompletionRequest(
            model=self.settings.model,
            messages=[ChatMessage(role=message.role, content=message.content) for message in messages],
            model_options=ChatModelOptions(
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            ),
            tool_config=ChatToolConfig(mode="none"),
        )
        try:
            response = self.client.chat.create(request)
        except httpx.TimeoutException as exc:
            raise ChatError("GigaChat не ответил вовремя. Попробуйте позже; история сохранена.") from exc
        except httpx.RequestError as exc:
            raise ChatError(
                "Не удалось подключиться к GigaChat. Проверьте интернет и доверенные "
                "сертификаты; при необходимости задайте GIGACHAT_CA_BUNDLE_FILE."
            ) from exc
        except ResponseError as exc:
            message = API_ERRORS.get(exc.status_code, "Ошибка сервиса GigaChat. Попробуйте позже.")
            raise ChatError(message) from exc
        except (GigaChatException, ValueError) as exc:
            # Тело ошибки SDK может содержать ключи и заголовки запроса.
            raise ChatError("Не удалось обработать ответ GigaChat. Попробуйте позже.") from exc
        return parse_reply(response)


def parse_reply(response: ChatCompletionResponse) -> Reply:
    """Собрать текст ответа и проверить, не прервалась ли генерация."""
    if response.error_details:
        raise ChatError("GigaChat сообщил об ошибке генерации. Попробуйте позже.")

    parts: list[str] = []
    truncated = response.finish_reason == "length"
    for message in response.messages:
        if message.function_call:
            raise ChatError("Модель запросила инструмент, но в этой версии доступны только текстовые ответы.")
        if message.finish_reason == "length":
            truncated = True
        for part in message.content or []:
            if part.function_call:
                raise ChatError("Модель запросила инструмент, но в этой версии доступны только текстовые ответы.")
            if message.role == "assistant" and part.text:
                parts.append(part.text)

    text = "\n".join(parts).strip()
    if not text:
        raise ChatError("Модель вернула пустой ответ.")
    return Reply(text, truncated=truncated)
