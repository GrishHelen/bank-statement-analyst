import pytest

from src.chat import ChatError, ChatSession, Message, Reply
from src.cli import run_chat


class FakeModel:
    def __init__(self) -> None:
        self.requests: list[tuple[Message, ...]] = []
        self.error: ChatError | None = None

    def complete(self, messages: list[Message]) -> Reply:
        self.requests.append(tuple(messages))
        if self.error:
            raise self.error
        return Reply("Ответ")


def test_follow_up_includes_previous_question_and_answer() -> None:
    model = FakeModel()
    chat = ChatSession(model.complete, "Правила")
    chat.ask("Первый вопрос")
    chat.ask("Уточнение")
    assert model.requests[-1] == (
        Message("system", "Правила"), Message("user", "Первый вопрос"),
        Message("assistant", "Ответ"), Message("user", "Уточнение"),
    )


def test_failed_turn_does_not_pollute_history() -> None:
    model = FakeModel()
    chat = ChatSession(model.complete, "Правила")
    chat.ask("Первый вопрос")
    previous = chat.history
    model.error = ChatError("Сервис недоступен")
    with pytest.raises(ChatError):
        chat.ask("Запрос со сбоем")
    assert chat.history == previous
    model.error = None
    chat.ask("Повтор")
    assert all(message.content != "Запрос со сбоем" for message in model.requests[-1])


def test_history_limit_keeps_complete_pairs_and_system_prompt() -> None:
    model = FakeModel()
    chat = ChatSession(model.complete, "Правила", max_history_turns=1)
    for question in ("Первый", "Второй", "Третий"):
        chat.ask(question)
    assert model.requests[-1] == (
        Message("system", "Правила"), Message("user", "Второй"),
        Message("assistant", "Ответ"), Message("user", "Третий"),
    )
    assert len(chat.history) == 2


@pytest.mark.parametrize("text", [" ", "Слишком длинное сообщение"])
def test_invalid_input_does_not_call_model(text: str) -> None:
    model = FakeModel()
    chat = ChatSession(model.complete, "Правила", max_input_chars=5)
    with pytest.raises(ChatError):
        chat.ask(text)
    assert not model.requests


def test_terminal_commands_and_reset_do_not_call_model() -> None:
    model = FakeModel()
    chat = ChatSession(model.complete, "Правила")
    inputs = iter(["", "/help", "Первый", "/reset", "/unknown", "Второй", "/exit"])
    output: list[str] = []

    def read(prompt: str) -> str:
        return next(inputs)

    assert run_chat(chat, read, output.append) == 0
    assert len(model.requests) == 2
    assert model.requests[-1] == (Message("system", "Правила"), Message("user", "Второй"))
    assert "История очищена." in output


def test_terminal_remains_usable_after_provider_error() -> None:
    model = FakeModel()
    model.error = ChatError("Сервис недоступен")
    inputs = iter(["Вопрос", "/help", "/exit"])
    output: list[str] = []

    def read(prompt: str) -> str:
        return next(inputs)

    assert run_chat(ChatSession(model.complete, "Правила"), read, output.append) == 0
    assert "Ошибка: Сервис недоступен" in output
    assert output[-1] == "До свидания!"
