"""Подключение к GigaChat: преобразование сообщений и вызовов функций в формат SDK."""

import json
from types import TracebackType
from typing import Any

import httpx
from gigachat import GigaChat
from gigachat.exceptions import GigaChatException, ResponseError
from gigachat.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatContentPart,
    ChatFunctionResult,
    ChatFunctionSpecification,
    ChatMessage,
    ChatModelOptions,
    ChatTool,
    ChatToolConfig,
)
from gigachat.models.chat_completions import ChatFunctionCall

from src.chat import ChatError, Message, Reply, ToolCall
from src.config import Settings
from src.providers.common import API_ERRORS, build_ssl_context


class GigaChatModel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            self.client = GigaChat(
                credentials=settings.credentials,
                scope=settings.scope,
                model=settings.model,
                base_url=settings.base_url,
                auth_url=settings.auth_url,
                timeout=settings.timeout,
                ssl_context=build_ssl_context(settings.ca_bundle_file),
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

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Reply:
        request = ChatCompletionRequest(
            model=self.settings.model,
            messages=to_gigachat_messages(messages),
            model_options=ChatModelOptions(
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            ),
            tool_config=ChatToolConfig(mode="auto" if tools else "none"),
        )
        if tools:
            request.tools = [ChatTool(functions={"specifications": [
                ChatFunctionSpecification(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool["parameters"],
                    few_shot_examples=tool.get("examples") or None,
                )
                for tool in tools
            ]})]
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


def to_gigachat_messages(messages: list[Message]) -> list[ChatMessage]:
    """Результаты инструментов одного шага объединяются в одно сообщение role=tool."""
    result: list[ChatMessage] = []
    for message in messages:
        if message.role == "tool":
            part = ChatContentPart(function_result=ChatFunctionResult(
                id_=message.tool_call_id, name=message.name or "", result=json.loads(message.content),
            ))
            previous = result[-1] if result else None
            if previous is not None and previous.role == "tool" and previous.tools_state_id == message.state:
                previous.content = (previous.content or []) + [part]
            else:
                result.append(ChatMessage(role="tool", tools_state_id=message.state, content=[part]))
        elif message.tool_calls:
            parts = [ChatContentPart(text=message.content)] if message.content else []
            for call in message.tool_calls:
                parts.append(ChatContentPart(function_call=ChatFunctionCall(
                    id_=call.id, name=call.name, arguments=call.arguments,
                )))
            result.append(ChatMessage(role="assistant", tools_state_id=message.state, content=parts))
        else:
            result.append(ChatMessage(role=message.role, content=message.content))
    return result


def parse_reply(response: ChatCompletionResponse) -> Reply:
    """Собрать текст и вызовы функций; проверить, не прервалась ли генерация."""
    if response.error_details:
        raise ChatError("GigaChat сообщил об ошибке генерации. Попробуйте позже.")

    parts: list[str] = []
    calls: list[ToolCall] = []
    state: str | None = None
    truncated = response.finish_reason == "length"
    for message in response.messages:
        if message.role != "assistant":
            continue
        if message.finish_reason == "length":
            truncated = True
        if message.tools_state_id:
            state = message.tools_state_id
        if message.function_call:
            calls.append(to_tool_call(message.function_call))
        for part in message.content or []:
            if part.function_call:
                calls.append(to_tool_call(part.function_call))
            elif part.text:
                parts.append(part.text)

    text = "\n".join(parts).strip()
    if not text and not calls:
        raise ChatError("Модель вернула пустой ответ.")
    return Reply(text, truncated=truncated, tool_calls=tuple(calls), state=state)


def to_tool_call(call: ChatFunctionCall) -> ToolCall:
    return ToolCall(id=call.id_, name=call.name, arguments=call.arguments)
