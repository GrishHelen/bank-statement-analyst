"""Проверки цикла агента с подменённой моделью: модель «выбирает» инструменты по сценарию."""

import json
from typing import Any

import pytest

from src.chat import ChatError, ChatSession, Message, Reply, ToolCall
from src.tools.base import Tool, ToolRegistry, ToolTrace
from src.tools.statement_tools import SearchArgs


class ScriptedModel:
    """Возвращает заранее заданные ответы и запоминает, что видела модель."""

    def __init__(self, replies: list[Reply]) -> None:
        self.replies = list(replies)
        self.requests: list[list[Message]] = []
        self.tools_seen: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Reply:
        self.requests.append(list(messages))
        self.tools_seen.append(tools or [])
        return self.replies.pop(0)


def call(name: str, arguments: Any, call_id: str = "c1") -> Reply:
    return Reply("", tool_calls=(ToolCall(call_id, name, arguments),), state="state")


def tool_results(messages: list[Message]) -> list[dict[str, Any]]:
    return [json.loads(m.content) for m in messages if m.role == "tool"]


def test_correct_calculation_reaches_model_and_user(tools: ToolRegistry) -> None:
    model = ScriptedModel([
        call("spending_summary", {"date_from": "2026-09-01", "date_to": "2026-09-30",
                                  "category": "кафе и рестораны"}),
        Reply("На кафе в сентябре ушло 2000 ₽."),
    ])
    steps: list[ToolTrace] = []
    session = ChatSession(model.complete, "Правила", tools=tools, on_step=steps.append)
    reply = session.ask("Сколько я потратил на кафе в сентябре?")

    assert reply.text == "На кафе в сентябре ушло 2000 ₽." and reply.steps == 2
    assert {spec["name"] for spec in model.tools_seen[0]} == {
        "search_transactions", "spending_summary", "find_recurring_payments"}
    result = tool_results(model.requests[1])[0]
    assert result["ok"] and result["result"]["by_currency"][0]["total"] == 2000.00
    assert steps[0].ok and "2000.00 RUB" in steps[0].summary
    # В истории только вопрос и итоговый ответ, без промежуточных шагов.
    assert [m.role for m in session.history] == ["user", "assistant"]


def test_invalid_arguments_are_returned_to_model_which_can_fix_them(tools: ToolRegistry) -> None:
    model = ScriptedModel([
        call("search_transactions", {"date_from": "2026-09-31", "limit": 100}),
        call("search_transactions", {"date_from": "2026-09-30", "limit": 5}, call_id="c2"),
        Reply("Вот операции."),
    ])
    session = ChatSession(model.complete, "Правила", tools=tools)
    assert session.ask("Покажи траты с 31 сентября").text == "Вот операции."
    first_error = tool_results(model.requests[1])[0]
    assert first_error["ok"] is False and first_error["error"]["code"] == "invalid_arguments"
    assert any("limit" in detail for detail in first_error["error"]["details"])
    assert tool_results(model.requests[2])[1]["ok"] is True


def test_empty_result_is_passed_as_data(tools: ToolRegistry) -> None:
    model = ScriptedModel([
        call("search_transactions", {"text": "Аэрофлот"}),
        Reply("Операций с Аэрофлотом в выписке нет."),
    ])
    session = ChatSession(model.complete, "Правила", tools=tools)
    session.ask("Были ли покупки в Аэрофлоте?")
    result = tool_results(model.requests[1])[0]
    assert result["ok"] and result["result"]["total_found"] == 0 and "note" in result["result"]


def test_tool_failure_does_not_crash_agent() -> None:
    def broken(args: SearchArgs) -> dict[str, Any]:
        raise RuntimeError("database is down: secret-connection-string")

    registry = ToolRegistry([Tool("search_transactions", "Поиск", "read", SearchArgs, broken)])
    model = ScriptedModel([call("search_transactions", {}), Reply("Не удалось получить операции.")])
    steps: list[ToolTrace] = []
    session = ChatSession(model.complete, "Правила", tools=registry, on_step=steps.append)

    assert session.ask("Покажи операции").text == "Не удалось получить операции."
    result = tool_results(model.requests[1])[0]
    assert result["error"]["code"] == "tool_failed"
    assert "secret" not in json.dumps(result)
    assert steps[0].ok is False


def test_unknown_tool_is_reported_to_model(tools: ToolRegistry) -> None:
    model = ScriptedModel([call("block_card", {"card": "1234"}), Reply("Я не могу блокировать карты.")])
    ChatSession(model.complete, "Правила", tools=tools).ask("Заблокируй карту")
    assert tool_results(model.requests[1])[0]["error"]["code"] == "unknown_tool"


def test_step_limit_stops_agent_and_keeps_history_clean(tools: ToolRegistry) -> None:
    model = ScriptedModel([call("search_transactions", {}, call_id=f"c{i}") for i in range(10)])
    session = ChatSession(model.complete, "Правила", tools=tools, max_steps=3)
    with pytest.raises(ChatError, match="лимит 3 шагов"):
        session.ask("Зацикленный вопрос")
    assert len(model.requests) == 3  # ровно max_steps обращений к модели
    assert session.history == ()


def test_several_calls_in_one_step_are_all_executed(tools: ToolRegistry) -> None:
    both = Reply("", tool_calls=(
        ToolCall("a", "spending_summary", {"date_from": "2026-09-01", "date_to": "2026-09-30"}),
        ToolCall("b", "find_recurring_payments", {}),
    ), state="s")
    model = ScriptedModel([both, Reply("Готово.")])
    ChatSession(model.complete, "Правила", tools=tools).ask("Сколько потратил в сентябре и какие подписки?")
    results = tool_results(model.requests[1])
    assert [r["tool"] for r in results] == ["spending_summary", "find_recurring_payments"]
    tool_messages = [m for m in model.requests[1] if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["a", "b"]
