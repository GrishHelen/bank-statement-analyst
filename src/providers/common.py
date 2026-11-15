"""Общее для всех провайдеров: SSL-контекст и тексты ошибок API."""

import ssl
from pathlib import Path

import certifi

API_ERRORS: dict[int, str] = {
    400: "Сервис отклонил запрос. Сократите сообщение или очистите историю командой /reset.",
    401: "Ошибка авторизации. Проверьте ключ и тип доступа в .env.",
    403: "Доступ запрещён. Проверьте права проекта, доступ к модели и остаток квоты.",
    404: "Модель или адрес API не найдены. Проверьте модель и адрес API в .env.",
    413: "Слишком большой запрос. Сократите сообщение или используйте /reset.",
    422: "Сервис не принял параметры запроса. Проверьте настройки модели.",
    429: "Достигнут лимит запросов или квота. Проверьте кабинет провайдера и повторите позже.",
}


def build_ssl_context(extra_ca_file: Path | None) -> ssl.SSLContext:
    """Проверка сертификатов всегда включена; можно добавить свой корневой сертификат."""
    context = ssl.create_default_context()
    context.load_verify_locations(certifi.where())
    if extra_ca_file:
        context.load_verify_locations(str(extra_ca_file))
    return context
