"""Ввод сообщений, загрузка выписки и вывод ответов в терминале."""

import argparse
import json
from pathlib import Path
from typing import Callable

from src.chat import ChatError, ChatSession, Message
from src.config import ConfigurationError, load_settings
from src.importers.csv_statement import StatementError, load_statement
from src.prompts import SYSTEM_PROMPT, with_statement_context
from src.tools.base import ToolTrace
from src.tools.statement_tools import StatementStore, make_tools

HELP: str = (
    "/load <путь к CSV> — загрузить выписку; /statement — что загружено; "
    "/help — справка; /reset — очистить историю; /exit — выйти."
)


def show_reply(session: ChatSession, text: str, write: Callable[[str], None]) -> bool:
    try:
        reply = session.ask(text)
    except ChatError as exc:
        write(f"Ошибка: {exc}")
        return False
    write(f"\nАгент: {reply.text}\n")
    if reply.truncated:
        write("Ответ достиг лимита длины. Можно попросить продолжить.")
    return True


def trace_printer(write: Callable[[str], None]) -> Callable[[ToolTrace], None]:
    def show(step: ToolTrace) -> None:
        arguments = step.arguments
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        mark = "✓" if step.ok else "✗"
        write(f"  [{mark} {step.name}] {arguments} → {step.summary}")
    return show


def load_into(session: ChatSession, store: StatementStore, path: str, write: Callable[[str], None]) -> bool:
    try:
        statement = load_statement(Path(path.strip().strip("\"'")))
    except StatementError as exc:
        write(f"Ошибка: {exc}")
        return False
    store.statement = statement
    session.system = Message("system", with_statement_context(SYSTEM_PROMPT, statement.describe()))
    session.reset()
    write(f"Загружено: {statement.describe()}. История очищена.")
    return True


def run_chat(
    session: ChatSession,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    store: StatementStore | None = None,
) -> int:
    store = store if store is not None else StatementStore()
    write("Аналитик банковской выписки. " + HELP)
    write(f"В памяти — последние {session.max_history_turns} пар вопрос–ответ; "
          f"не больше {session.max_steps} шагов агента на вопрос.")
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
            elif command == "/load" or command.startswith("/load "):
                path = text[len("/load"):].strip()
                if path:
                    load_into(session, store, path, write)
                else:
                    write("Укажите путь: /load data/sample_statement.csv")
            elif command == "/statement":
                write(store.statement.describe() if store.statement else "Выписка не загружена.")
            elif command.startswith("/"):
                write("Неизвестная команда. " + HELP)
            else:
                show_reply(session, text, write)
        except (EOFError, KeyboardInterrupt):
            break
    write("До свидания!")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Агент для анализа банковской выписки")
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"),
        help="Файл настроек (по умолчанию .env в текущей папке)",
    )
    parser.add_argument("--statement", help="CSV-выписка, которую загрузить при запуске")
    parser.add_argument("--message", help="Отправить одно сообщение и завершить работу")
    parser.add_argument("--quiet", action="store_true", help="Не печатать шаги агента")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.env_file)
        from src.providers import create_model, model_label

        store = StatementStore()
        with create_model(settings) as model:
            session = ChatSession(
                model.complete, with_statement_context(SYSTEM_PROMPT, None),
                max_history_turns=settings.max_history_turns,
                max_input_chars=settings.max_input_chars,
                tools=make_tools(store),
                max_steps=settings.max_agent_steps,
                on_step=None if args.quiet else trace_printer(print),
            )
            print(f"Модель: {model_label(settings)}")
            if args.statement and not load_into(session, store, args.statement, print):
                return 1
            if args.message is not None:
                return 0 if show_reply(session, args.message, print) else 1
            return run_chat(session, store=store)
    except (ConfigurationError, ChatError) as exc:
        print(f"Ошибка: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nРабота завершена.")
        return 130
