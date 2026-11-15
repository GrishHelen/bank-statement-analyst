"""Любая модель с OpenAI-совместимым API: Z.ai GLM, Groq, OpenRouter, Ollama и другие.

Провайдер выбирается в .env (LLM_PROVIDER=openai), адрес, ключ и модель — там же.
"""

import json
from types import TracebackType
from typing import Any

import httpx
import openai

from src.chat import ChatError, Message, Reply, ToolCall
from src.config import Settings
from src.providers.common import API_ERRORS, build_ssl_context


class OpenAICompatibleModel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            self.http = httpx.Client(verify=build_ssl_context(settings.ca_bundle_file), timeout=settings.timeout)
            self.client = openai.OpenAI(
                api_key=settings.openai_api_key or "not-needed",
                base_url=settings.openai_base_url,
                timeout=settings.timeout,
                max_retries=0,
                http_client=self.http,
            )
        except (OSError, ValueError) as exc:
            raise ChatError("Не удалось настроить OpenAI-совместимый клиент. Проверьте настройки.") from exc

    def __enter__(self) -> "OpenAICompatibleModel":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.client.close()
        self.http.close()

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Reply:
        options: dict[str, Any] = {}
        if tools:
            options["tools"] = [
                {"type": "function", "function": {
                    "name": tool["name"], "description": tool["description"], "parameters": tool["parameters"],
                }}
                for tool in tools
            ]
            options["tool_choice"] = "auto"
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=to_openai_messages(messages),
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                **options,
            )
        except openai.APITimeoutError as exc:
            raise ChatError("Модель не ответила вовремя. Попробуйте позже; история сохранена.") from exc
        except openai.APIConnectionError as exc:
            raise ChatError("Не удалось подключиться к API модели. Проверьте OPENAI_BASE_URL и интернет.") from exc
        except openai.APIStatusError as exc:
            raise ChatError(API_ERRORS.get(exc.status_code, "Ошибка сервиса модели. Попробуйте позже.")) from exc
        except (openai.OpenAIError, ValueError) as exc:
            raise ChatError("Не удалось обработать ответ модели. Попробуйте позже.") from exc
        return parse_reply(response)


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            result.append({"role": "tool", "tool_call_id": message.tool_call_id or "", "content": message.content})
        elif message.tool_calls:
            result.append({
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": [
                    {"id": call.id or f"call_{index}", "type": "function", "function": {
                        "name": call.name,
                        "arguments": call.arguments if isinstance(call.arguments, str)
                        else json.dumps(call.arguments, ensure_ascii=False),
                    }}
                    for index, call in enumerate(message.tool_calls)
                ],
            })
        else:
            result.append({"role": message.role, "content": message.content})
    return result


def parse_reply(response: Any) -> Reply:
    if not response.choices:
        raise ChatError("Модель вернула пустой ответ.")
    choice = response.choices[0]
    message = choice.message
    calls = tuple(
        ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
        for call in (message.tool_calls or [])
    )
    text = (message.content or "").strip()
    if not text and not calls:
        raise ChatError("Модель вернула пустой ответ.")
    return Reply(text, truncated=choice.finish_reason == "length", tool_calls=calls)
