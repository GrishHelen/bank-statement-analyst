"""Загрузка CSV-выписки с проверкой каждой строки.

Формат: UTF-8, первая строка — заголовок, разделитель «,» или «;».
Обязательные столбцы: operation_id, date, description, amount, currency, mcc.
Сумма со знаком: расход отрицательный, поступление положительное.
"""

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.importers.mcc import category_for

REQUIRED_COLUMNS: tuple[str, ...] = ("operation_id", "date", "description", "amount", "currency", "mcc")
SUPPORTED_CURRENCIES: frozenset[str] = frozenset({"RUB", "USD", "EUR", "CNY"})
MAX_FILE_BYTES: int = 5 * 1024 * 1024
MAX_ROWS: int = 20000
MAX_DESCRIPTION_CHARS: int = 200
MAX_REPORTED_ERRORS: int = 5


class StatementError(Exception):
    """Выписку нельзя загрузить; текст можно показать пользователю."""


@dataclass(frozen=True)
class Transaction:
    operation_id: str
    date: date
    description: str
    amount: Decimal
    currency: str
    mcc: str
    category: str

    @property
    def is_expense(self) -> bool:
        return self.amount < 0


@dataclass(frozen=True)
class Statement:
    source: str
    transactions: tuple[Transaction, ...]

    @property
    def date_from(self) -> date:
        return min(t.date for t in self.transactions)

    @property
    def date_to(self) -> date:
        return max(t.date for t in self.transactions)

    @property
    def currencies(self) -> list[str]:
        return sorted({t.currency for t in self.transactions})

    def describe(self) -> str:
        return (
            f"{self.source}: {len(self.transactions)} операций "
            f"с {self.date_from.isoformat()} по {self.date_to.isoformat()}, "
            f"валюты: {', '.join(self.currencies)}"
        )


def load_statement(path: Path) -> Statement:
    path = path.expanduser()
    if not path.is_file():
        raise StatementError(f"Файл выписки не найден: {path}")
    if path.suffix.lower() != ".csv":
        raise StatementError("Поддерживаются только CSV-файлы.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise StatementError("Файл больше 5 МБ. Разделите выписку на части.")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise StatementError("Выписка должна быть в кодировке UTF-8.") from exc
    return parse_statement(text, source=path.name)


def parse_statement(text: str, source: str = "выписка") -> Statement:
    lines = text.splitlines()
    if not lines:
        raise StatementError("Файл выписки пустой.")
    delimiter = ";" if lines[0].count(";") > lines[0].count(",") else ","
    reader = csv.DictReader(lines, delimiter=delimiter)
    header = [name.strip() for name in reader.fieldnames or []]
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise StatementError("В выписке нет столбцов: " + ", ".join(missing) + ".")
    reader.fieldnames = header

    transactions: list[Transaction] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(reader, start=2):
        if len(transactions) >= MAX_ROWS:
            raise StatementError(f"В выписке больше {MAX_ROWS} операций.")
        if not any((value or "").strip() for value in row.values() if isinstance(value, str)):
            continue
        try:
            transaction = parse_row(row, delimiter)
        except ValueError as exc:
            errors.append(f"строка {line_number}: {exc}")
            continue
        if transaction.operation_id in seen_ids:
            errors.append(f"строка {line_number}: повторяется operation_id {transaction.operation_id}")
            continue
        seen_ids.add(transaction.operation_id)
        transactions.append(transaction)

    if errors:
        shown = "; ".join(errors[:MAX_REPORTED_ERRORS])
        more = f" и ещё {len(errors) - MAX_REPORTED_ERRORS}" if len(errors) > MAX_REPORTED_ERRORS else ""
        raise StatementError(f"Выписка не загружена, ошибки в данных: {shown}{more}.")
    if not transactions:
        raise StatementError("В выписке нет операций.")
    transactions.sort(key=lambda t: (t.date, t.operation_id))
    return Statement(source=source, transactions=tuple(transactions))


def parse_row(row: dict[str, str], delimiter: str) -> Transaction:
    def field(name: str) -> str:
        return (row.get(name) or "").strip()

    operation_id = field("operation_id")
    if not operation_id or len(operation_id) > 64:
        raise ValueError("пустой или слишком длинный operation_id")
    try:
        day = date.fromisoformat(field("date"))
    except ValueError as exc:
        raise ValueError(f"дата «{field('date')}» не в формате ГГГГ-ММ-ДД") from exc
    description = " ".join(field("description").split())
    if not description:
        raise ValueError("пустое описание")
    description = description[:MAX_DESCRIPTION_CHARS]
    raw_amount = field("amount").replace(" ", "").replace(" ", "")
    if delimiter == ";":
        raw_amount = raw_amount.replace(",", ".")
    try:
        amount = Decimal(raw_amount)
    except InvalidOperation as exc:
        raise ValueError(f"сумма «{field('amount')}» не является числом") from exc
    if not amount.is_finite() or amount == 0:
        raise ValueError("сумма должна быть ненулевым конечным числом")
    if amount.as_tuple().exponent < -2:  # type: ignore[operator]
        raise ValueError("в сумме больше двух знаков после точки")
    currency = field("currency").upper()
    if currency not in SUPPORTED_CURRENCIES:
        raise ValueError(f"валюта «{currency}» не поддерживается")
    mcc = field("mcc")
    if mcc and not (len(mcc) == 4 and mcc.isdigit()):
        raise ValueError(f"MCC «{mcc}» должен состоять из 4 цифр")
    return Transaction(
        operation_id=operation_id,
        date=day,
        description=description,
        amount=amount,
        currency=currency,
        mcc=mcc,
        category=category_for(mcc, amount > 0),
    )
