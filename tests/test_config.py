import os
from pathlib import Path

import pytest

from src.config import ConfigurationError, load_settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("GIGACHAT_", "APP_MAX_")):
            monkeypatch.delenv(name)


def test_missing_key_is_reported_without_secret(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GIGACHAT_CREDENTIALS=\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="GIGACHAT_CREDENTIALS"):
        load_settings(env)


def test_environment_overrides_file_and_secret_is_not_in_repr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = tmp_path / ".env"
    env.write_text("GIGACHAT_CREDENTIALS=file-secret\nGIGACHAT_MODEL=GigaChat\n", encoding="utf-8")
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "environment-secret")
    settings = load_settings(env)
    assert settings.credentials == "environment-secret"
    assert "environment-secret" not in repr(settings)


@pytest.mark.parametrize("setting", [
    "GIGACHAT_TIMEOUT=nan",
    "GIGACHAT_TIMEOUT=inf",
    "APP_MAX_HISTORY_TURNS=0",
    "GIGACHAT_MAX_TOKENS=wrong",
    "GIGACHAT_MAX_TOKENS=1.5",
    "GIGACHAT_BASE_URL=http://example.com",
])
def test_invalid_configuration_is_rejected(tmp_path: Path, setting: str) -> None:
    env = tmp_path / ".env"
    env.write_text("GIGACHAT_CREDENTIALS=test-only\n" + setting, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_settings(env)


def test_certificate_path_is_relative_to_env_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    cert = tmp_path / "ca.pem"
    cert.write_text("test-only", encoding="utf-8")
    env.write_text("GIGACHAT_CREDENTIALS=test-only\nGIGACHAT_CA_BUNDLE_FILE=ca.pem", encoding="utf-8")
    assert load_settings(env).ca_bundle_file == cert.resolve()
