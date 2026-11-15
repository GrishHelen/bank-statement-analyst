from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from gigachat.exceptions import ResponseError
from gigachat.models import ChatCompletionResponse, ChatContentPart, ChatMessage
from gigachat.models.chat_completions import ChatFunctionCall

from src.chat import ChatError, Message, ToolCall
from src.cli import main
from src.config import Settings
from src.providers.gigachat import GigaChatModel, parse_reply


def make_provider(monkeypatch: pytest.MonkeyPatch) -> tuple[GigaChatModel, Mock, Mock]:
    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr("src.providers.gigachat.GigaChat", constructor)
    return GigaChatModel(Settings(credentials="test-only")), client, constructor


def test_adapter_sends_history_limits_and_disables_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, client, constructor = make_provider(monkeypatch)
    client.chat.create.return_value = ChatCompletionResponse(
        messages=[ChatMessage(role="assistant", content="Привет!", finish_reason="length")]
    )
    with provider:
        reply = provider.complete([Message("system", "Правила"), Message("user", "Вопрос")])
    request = client.chat.create.call_args.args[0]
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.model_options.max_tokens == 1024
    assert request.tool_config.mode == "none"
    assert reply.text == "Привет!" and reply.truncated
    assert constructor.call_args.kwargs["ssl_context"].check_hostname
    client.close.assert_called_once()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422, 429, 503])
def test_provider_errors_never_expose_headers_or_body(
    monkeypatch: pytest.MonkeyPatch, status: int,
) -> None:
    provider, client, _ = make_provider(monkeypatch)
    client.chat.create.side_effect = ResponseError(
        "https://example.com", status, b"secret-in-body", httpx.Headers({"Authorization": "secret-in-header"})
    )
    with pytest.raises(ChatError) as caught:
        provider.complete([Message("user", "Вопрос")])
    assert "secret-in" not in str(caught.value)


def test_timeout_is_a_user_facing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, client, _ = make_provider(monkeypatch)
    client.chat.create.side_effect = httpx.ReadTimeout("technical-detail")
    with pytest.raises(ChatError, match="не ответил вовремя"):
        provider.complete([Message("user", "Вопрос")])


def test_empty_response_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, client, _ = make_provider(monkeypatch)
    client.chat.create.return_value = ChatCompletionResponse(messages=[])
    with pytest.raises(ChatError, match="пустой ответ"):
        provider.complete([Message("user", "Вопрос")])


def test_reply_joins_assistant_text_and_keeps_truncation() -> None:
    response = ChatCompletionResponse(
        messages=[
            ChatMessage(role="user", content="Не включать в ответ"),
            ChatMessage(role="assistant", content=[
                ChatContentPart(text="Первая часть"),
                ChatContentPart(text="Вторая часть"),
            ]),
        ],
        finish_reason="length",
    )
    reply = parse_reply(response)
    assert reply.text == "Первая часть\nВторая часть"
    assert reply.truncated


@pytest.mark.parametrize("nested", [False, True])
def test_tool_call_is_parsed_from_both_formats(nested: bool) -> None:
    call = ChatFunctionCall(id="call-1", name="test_tool", arguments={"limit": 3})
    if nested:
        message = ChatMessage(role="assistant", content=[ChatContentPart(function_call=call)],
                              tools_state_id="state-1")
    else:
        message = ChatMessage(role="assistant", function_call=call, tools_state_id="state-1")
    reply = parse_reply(ChatCompletionResponse(messages=[message]))
    assert reply.tool_calls == (ToolCall("call-1", "test_tool", {"limit": 3}),)
    assert reply.state == "state-1"


def test_adapter_sends_tools_and_returns_results_with_state(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, client, _ = make_provider(monkeypatch)
    client.chat.create.return_value = ChatCompletionResponse(
        messages=[ChatMessage(role="assistant", content="Итого 100 ₽")]
    )
    tool = {"name": "spending_summary", "description": "Сумма", "examples": [],
            "parameters": {"type": "object", "properties": {}}}
    history = [
        Message("user", "Сколько?"),
        Message("assistant", "", tool_calls=(ToolCall("c1", "spending_summary", {}),), state="s1"),
        Message("tool", '{"ok": true}', tool_call_id="c1", name="spending_summary", state="s1"),
    ]
    provider.complete(history, [tool])
    request = client.chat.create.call_args.args[0]
    assert request.tool_config.mode == "auto"
    assert request.tools[0].functions.specifications[0].name == "spending_summary"
    assistant, result = request.messages[1], request.messages[2]
    assert assistant.tools_state_id == "s1" and assistant.content[0].function_call.name == "spending_summary"
    assert result.role == "tool" and result.tools_state_id == "s1"
    assert result.content[0].function_result.result == {"ok": True}


def test_single_message_runs_through_chat_and_mock_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = tmp_path / ".env"
    env.write_text("GIGACHAT_CREDENTIALS=test-only\n", encoding="utf-8")
    client = Mock()
    client.chat.create.return_value = ChatCompletionResponse(
        messages=[ChatMessage(role="assistant", content="Тестовый ответ")],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.providers.gigachat.GigaChat", Mock(return_value=client))

    assert main(["--env-file", str(env), "--message", "Тестовый вопрос"]) == 0
    assert "Тестовый ответ" in capsys.readouterr().out
    client.chat.create.assert_called_once()
    client.close.assert_called_once()
