"""История диалога и один запрос к модели на каждое сообщение."""

from dataclasses import dataclass
from typing import Callable, Literal


class ChatError(Exception):
    """Ошибка с текстом, который можно показать пользователю."""


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Reply:
    text: str
    truncated: bool = False


class ChatSession:
    def __init__(
        self,
        ask_model: Callable[[list[Message]], Reply],
        system_prompt: str,
        max_history_turns: int = 10,
        max_input_chars: int = 4000,
    ) -> None:
        self.ask_model = ask_model
        self.system = Message("system", system_prompt)
        self.max_history_turns = max_history_turns
        self.max_input_chars = max_input_chars
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
        reply = self.ask_model(messages)
        if not reply.text.strip():
            raise ChatError("Модель вернула пустой ответ. Попробуйте другой вопрос.")
        # Сохраняем пару сообщений только после успешного ответа.
        self._history.append(user_message)
        self._history.append(Message("assistant", reply.text))
        self._history = self._history[-2 * self.max_history_turns:]
        return reply
