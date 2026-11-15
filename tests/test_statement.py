from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from src.importers.csv_statement import StatementError, load_statement, parse_statement
from src.tools.statement_tools import StatementStore, make_tools

HEADER = "operation_id,date,description,amount,currency,mcc\n"


def test_rows_are_normalized_and_categorized(statement) -> None:
    first = statement.transactions[0]
    assert first.date == date(2026, 8, 5)
    assert first.amount == Decimal("-169.00")
    assert first.category == "цифровые подписки"
    salary = next(t for t in statement.transactions if t.description == "Зарплата")
    assert salary.category == "доходы" and not salary.is_expense
    assert statement.currencies == ["EUR", "RUB"]


def test_semicolon_and_decimal_comma_are_supported() -> None:
    text = "operation_id;date;description;amount;currency;mcc\nA;2026-01-02;Кафе;-1 250,50;rub;5814\n"
    transaction = parse_statement(text).transactions[0]
    assert transaction.amount == Decimal("-1250.50") and transaction.currency == "RUB"


@pytest.mark.parametrize("row, problem", [
    ("1,2026-02-30,Кафе,-10,RUB,5814", "ГГГГ-ММ-ДД"),
    ("1,2026-01-01,Кафе,десять,RUB,5814", "не является числом"),
    ("1,2026-01-01,Кафе,0,RUB,5814", "ненулевым"),
    ("1,2026-01-01,Кафе,-10,BTC,5814", "не поддерживается"),
    ("1,2026-01-01,Кафе,-10,RUB,58", "4 цифр"),
    ("1,2026-01-01,,-10,RUB,5814", "пустое описание"),
])
def test_bad_rows_are_rejected_with_line_number(row: str, problem: str) -> None:
    with pytest.raises(StatementError, match=problem) as caught:
        parse_statement(HEADER + row)
    assert "строка 2" in str(caught.value)


def test_missing_columns_and_duplicate_ids_are_rejected() -> None:
    with pytest.raises(StatementError, match="currency"):
        parse_statement("operation_id,date,description,amount,mcc\n1,2026-01-01,Кафе,-10,5814")
    with pytest.raises(StatementError, match="повторяется"):
        parse_statement(HEADER + "1,2026-01-01,Кафе,-10,RUB,5814\n1,2026-01-02,Кафе,-10,RUB,5814")


def test_load_checks_file(tmp_path: Path) -> None:
    with pytest.raises(StatementError, match="не найден"):
        load_statement(tmp_path / "нет.csv")
    other = tmp_path / "выписка.pdf"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(StatementError, match="CSV"):
        load_statement(other)


def test_sample_statement_from_repository_loads_and_answers() -> None:
    path = Path(__file__).resolve().parent.parent / "data" / "sample_statement.csv"
    statement = load_statement(path)
    assert len(statement.transactions) == 207
    tools = make_tools(StatementStore(statement))
    result = tools.execute("find_recurring_payments", {})["result"]
    names = {payment["description"] for payment in result["payments"]}
    assert {"Яндекс Плюс подписка", "Онлайн-кинотеатр Иви", "Фитнес-клуб «Пульс», абонемент"} <= names
    assert "Пятёрочка" not in names
    ozon = tools.execute("search_transactions", {"text": "Озон", "date_from": "2026-10-14", "date_to": "2026-10-14"})
    assert ozon["result"]["total_found"] == 2  # двойное списание находится и по-русски
