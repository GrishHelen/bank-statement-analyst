"""Диалог и цикл агента: модель → инструмент → результат модели → ответ.

На каждый вопрос пользователя harness делает не больше max_steps обращений
к модели. Модель решает, вызвать ли инструмент и с какими аргументами;
код проверяет аргументы, выполняет функцию и возвращает результат модели.
В историю диалога попадают только вопрос и итоговый ответ, без промежуточных шагов.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from src.tools.base import ToolRegistry, ToolTrace, error_result


class ChatError(Exception):
    """Ошибка с текстом, который можно показать пользователю."""


@dataclass(frozen=True)
class ToolCall:
    id: Optional[str]
    name: str
    arguments: Any


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    # Служебный идентификатор провайдера (например, tools_state_id в GigaChat).
    state: Optional[str] = None


@dataclass(frozen=True)
class Reply:
    text: str
    truncated: bool = False
    tool_calls: tuple[ToolCall, ...] = ()
    state: Optional[str] = None
    trace: tuple[ToolTrace, ...] = field(default=())
    steps: int = 1


AskModel = Callable[[list[Message], list[dict[str, Any]]], Reply]


class ChatSession:
    def __init__(
        self,
        ask_model: AskModel,
        system_prompt: str,
        max_history_turns: int = 10,
        max_input_chars: int = 4000,
        tools: ToolRegistry | None = None,
        max_steps: int = 5,
        on_step: Callable[[ToolTrace], None] | None = None,
    ) -> None:
        self.ask_model = ask_model
        self.system = Message("system", system_prompt)
        self.max_history_turns = max_history_turns
        self.max_input_chars = max_input_chars
        self.tools = tools
        self.max_steps = max_steps
        self.on_step = on_step
        self._history: list[Message] = []

    @property
    def history(self) -> tuple[Message, ...]:
        return tuple(self._history)

    def reset(self) -> None:
        self._history.clear()

    def ask(self, text: str) -> Reply:
        text = text.strip()
        if not text:
            raise ChatError("Введите непустое сообщение.")
        if len(text) > self.max_input_chars:
            raise ChatError(f"Сократите сообщение до {self.max_input_chars} символов.")
        user_message = Message("user", text)
        messages = [self.system] + self._history + [user_message]
        specifications = self.tools.specifications() if self.tools else []
        trace: list[ToolTrace] = []

        for step in range(1, self.max_steps + 1):
            reply = self.ask_model(messages, specifications)
            if not reply.tool_calls:
                if not reply.text.strip():
                    raise ChatError("Модель вернула пустой ответ. Попробуйте другой вопрос.")
                # Сохраняем пару сообщений только после успешного ответа.
                self._history.append(user_message)
                self._history.append(Message("assistant", reply.text))
                self._history = self._history[-2 * self.max_history_turns:]
                return Reply(reply.text, truncated=reply.truncated, trace=tuple(trace), steps=step)
            if step == self.max_steps:
                break
            messages.append(Message("assistant", reply.text, tool_calls=reply.tool_calls, state=reply.state))
            for call in reply.tool_calls:
                result = self.run_tool(call)
                record = ToolTrace(call.name, call.arguments, bool(result.get("ok")), summarize(result))
                trace.append(record)
                if self.on_step:
                    self.on_step(record)
                messages.append(Message(
                    "tool", json.dumps(result, ensure_ascii=False),
                    tool_call_id=call.id, name=call.name, state=reply.state,
                ))

        raise ChatError(
            f"Агент не уложился в лимит {self.max_steps} шагов и остановлен. "
            "Уточните вопрос или разбейте его на части."
        )

    def run_tool(self, call: ToolCall) -> dict[str, Any]:
        if self.tools is None:
            return error_result(call.name, "unknown_tool", "Инструменты в этом режиме недоступны.")
        return self.tools.execute(call.name, call.arguments)


def summarize(result: dict[str, Any]) -> str:
    """Короткая строка о результате инструмента для журнала шагов."""
    if not result.get("ok"):
        error = result.get("error", {})
        details = "; ".join(error.get("details", [])[:2])
        return f"ошибка {error.get('code')}: {error.get('message')}" + (f" ({details})" if details else "")
    data = result.get("result", {})
    if "total_found" in data:
        return f"найдено {data['total_found']}, передано модели {data['returned']}"
    if "by_currency" in data:
        totals = ", ".join(f"{c['total']:.2f} {c['currency']}" for c in data["by_currency"])
        return "итого " + (totals or "0")
    if "payments" in data:
        return f"регулярных платежей: {len(data['payments'])}"
    return "готово"
