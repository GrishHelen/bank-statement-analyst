"""Ввод сообщений и вывод ответов в терминале."""

import argparse
from pathlib import Path
from typing import Callable

from src.chat import ChatError, ChatSession
from src.config import ConfigurationError, load_settings
from src.prompts import SYSTEM_PROMPT

HELP: str = "/help — справка; /reset — очистить историю; /exit — выйти."


def show_reply(session: ChatSession, text: str, write: Callable[[str], None]) -> bool:
    try:
        reply = session.ask(text)
    except ChatError as exc:
        write(f"Ошибка: {exc}")
        return False
    write(f"\nGigaChat: {reply.text}\n")
    if reply.truncated:
        write("Ответ достиг лимита длины. Можно попросить продолжить.")
    return True


def run_chat(
    session: ChatSession,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    write("Консольный чат с GigaChat. " + HELP)
    write(f"В памяти — последние {session.max_history_turns} пар вопрос–ответ. Выход очищает историю.")
    while True:
        try:
            text = read("Вы: ").strip()
            if not text:
                continue
            command = text.lower()
            if command == "/exit":
                break
            if command == "/reset":
                session.reset()
                write("История очищена.")
            elif command == "/help":
                write(HELP)
            elif command.startswith("/"):
                write("Неизвестная команда. " + HELP)
            else:
                show_reply(session, text, write)
        except (EOFError, KeyboardInterrupt):
            break
    write("До свидания!")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Консольный чат с GigaChat")
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"),
        help="Файл настроек (по умолчанию .env в текущей папке)",
    )
    parser.add_argument("--message", help="Отправить одно сообщение и завершить работу")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.env_file)
        from src.providers.gigachat import GigaChatModel

        with GigaChatModel(settings) as model:
            session = ChatSession(
                model.complete, SYSTEM_PROMPT,
                max_history_turns=settings.max_history_turns,
                max_input_chars=settings.max_input_chars,
            )
            if args.message is not None:
                return 0 if show_reply(session, args.message, print) else 1
            return run_chat(session)
    except (ConfigurationError, ChatError) as exc:
        print(f"Ошибка: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nРабота завершена.")
        return 130
