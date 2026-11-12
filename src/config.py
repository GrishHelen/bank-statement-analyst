"""Чтение настроек из .env и проверка их значений."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values


class ConfigurationError(Exception):
    """Ошибка в настройках без вывода ключей и токенов."""


@dataclass(frozen=True)
class Settings:
    credentials: str = field(repr=False)
    scope: str = "GIGACHAT_API_PERS"
    model: str = "GigaChat-2"
    base_url: str = "https://api.giga.chat/v1/"
    auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    timeout: float = 10.0
    max_tokens: int = 1024
    temperature: float = 0.3
    ca_bundle_file: Path | None = None
    max_history_turns: int = 5
    max_input_chars: int = 4000


def load_settings(env_file: Path) -> Settings:
    env_file = env_file.resolve()
    if not env_file.is_file():
        raise ConfigurationError("Файл настроек не найден. Создайте .env по образцу .env.example.")
    try:
        local = dotenv_values(env_file, encoding="utf-8-sig", interpolate=False)
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError("Не удалось прочитать .env как UTF-8.") from exc

    def value(name: str, default: str = "") -> str:
        return (os.environ.get(name, local.get(name)) or default).strip()

    def integer(name: str, default: int, low: int, high: int) -> int:
        try:
            parsed = int(value(name, str(default)))
        except ValueError as exc:
            raise ConfigurationError(f"В {name} должно быть целое число.") from exc
        if not low <= parsed <= high:
            raise ConfigurationError(f"{name} должен быть в диапазоне {low}–{high}.")
        return parsed

    def decimal(name: str, default: float, low: float, high: float) -> float:
        try:
            parsed = float(value(name, str(default)))
        except ValueError as exc:
            raise ConfigurationError(f"В {name} должно быть число.") from exc
        if not low <= parsed <= high:
            raise ConfigurationError(f"{name} должен быть в диапазоне {low:g}–{high:g}.")
        return parsed

    credentials = value("GIGACHAT_CREDENTIALS")
    if not credentials or credentials.startswith("<"):
        raise ConfigurationError("Заполните GIGACHAT_CREDENTIALS в локальном .env.")
    scope = value("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    if scope not in {"GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"}:
        raise ConfigurationError("Неизвестное значение GIGACHAT_SCOPE.")
    base_url = value("GIGACHAT_BASE_URL", Settings.base_url)
    auth_url = value("GIGACHAT_AUTH_URL", Settings.auth_url)
    for name, url in (("GIGACHAT_BASE_URL", base_url), ("GIGACHAT_AUTH_URL", auth_url)):
        try:
            parsed = urlsplit(url)
            valid = (
                parsed.scheme == "https"
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
            )
        except ValueError:
            valid = False
        if not valid:
            raise ConfigurationError(f"{name} должен содержать HTTPS-адрес без учётных данных.")
    ca_path = value("GIGACHAT_CA_BUNDLE_FILE")
    ca_bundle: Path | None = None
    if ca_path:
        ca_bundle = Path(ca_path).expanduser()
        if not ca_bundle.is_absolute():
            ca_bundle = env_file.parent / ca_bundle
        if not ca_bundle.is_file():
            raise ConfigurationError("Файл сертификатов GIGACHAT_CA_BUNDLE_FILE не найден.")
        ca_bundle = ca_bundle.resolve()
    return Settings(
        credentials=credentials,
        scope=scope,
        model=value("GIGACHAT_MODEL", Settings.model),
        base_url=base_url,
        auth_url=auth_url,
        ca_bundle_file=ca_bundle,
        timeout=decimal("GIGACHAT_TIMEOUT", 10.0, 1.0, 120.0),
        max_tokens=integer("GIGACHAT_MAX_TOKENS", 1024, 1, 4096),
        temperature=decimal("GIGACHAT_TEMPERATURE", 0.3, 0.0, 2.0),
        max_history_turns=integer("APP_MAX_HISTORY_TURNS", 5, 1, 50),
        max_input_chars=integer("APP_MAX_INPUT_CHARS", 4000, 1, 16000),
    )
