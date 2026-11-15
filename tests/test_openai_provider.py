import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.chat import Message, ToolCall
from src.config import ConfigurationError, Settings, load_settings
from src.providers import create_model
from src.providers.openai_compatible import OpenAICompatibleModel, to_openai_messages


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for name in list(os.environ):
        if name.startswith(("GIGACHAT_", "APP_", "LLM_", "OPENAI_")):
            monkeypatch.delenv(name)


def openai_settings() -> Settings:
    return Settings(provider="openai", openai_base_url="https://api.example.com/v1", openai_model="glm-test")


def test_provider_is_chosen_by_config(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "LLM_PROVIDER=openai\nOPENAI_BASE_URL=http://localhost:11434/v1\nOPENAI_MODEL=qwen3:8b\n",
        encoding="utf-8",
    )
    settings = load_settings(env)  # ключ GigaChat не нужен для другого провайдера
    assert settings.provider == "openai" and settings.openai_model == "qwen3:8b"
    with create_model(settings) as model:
        assert isinstance(model, OpenAICompatibleModel)


@pytest.mark.parametrize("content", [
    "LLM_PROVIDER=yandex\nGIGACHAT_CREDENTIALS=x\n",
    "LLM_PROVIDER=openai\nOPENAI_MODEL=m\n",
    "LLM_PROVIDER=openai\nOPENAI_BASE_URL=http://example.com/v1\nOPENAI_MODEL=m\n",
])
def test_bad_provider_settings_are_rejected(tmp_path: Path, content: str) -> None:
    env = tmp_path / ".env"
    env.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_settings(env)


def test_messages_and_tools_use_openai_format(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    tool_call = SimpleNamespace(id="call_1", function=SimpleNamespace(
        name="spending_summary", arguments='{"group_by": "month"}'))
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls", message=SimpleNamespace(content=None, tool_calls=[tool_call]))])
    monkeypatch.setattr("src.providers.openai_compatible.openai.OpenAI", Mock(return_value=client))
    model = OpenAICompatibleModel(openai_settings())
    tool = {"name": "spending_summary", "description": "Сумма", "parameters": {"type": "object"}}

    reply = model.complete([Message("user", "Сколько?")], [tool])

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "glm-test" and kwargs["tool_choice"] == "auto"
    assert kwargs["tools"][0]["function"]["name"] == "spending_summary"
    assert reply.tool_calls == (ToolCall("call_1", "spending_summary", '{"group_by": "month"}'),)


def test_tool_round_is_converted_for_openai() -> None:
    messages = to_openai_messages([
        Message("assistant", "", tool_calls=(ToolCall("call_1", "search_transactions", {"limit": 2}),)),
        Message("tool", '{"ok": true}', tool_call_id="call_1", name="search_transactions"),
    ])
    assert json.loads(messages[0]["tool_calls"][0]["function"]["arguments"]) == {"limit": 2}
    assert messages[1] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}
