from src.tools.base import ToolRegistry
from src.tools.statement_tools import StatementStore, make_tools


def test_summary_counts_september_expenses_by_category_and_currency(tools: ToolRegistry) -> None:
    answer = tools.execute("spending_summary", {"date_from": "2026-09-01", "date_to": "2026-09-30"})
    assert answer["ok"]
    by_currency = {entry["currency"]: entry for entry in answer["result"]["by_currency"]}
    assert by_currency["RUB"]["total"] == 3169.00
    groups = {group["group"]: group for group in by_currency["RUB"]["groups"]}
    assert groups["кафе и рестораны"]["total"] == 2000.00 and groups["кафе и рестораны"]["count"] == 2
    assert groups["продукты"]["share_pct"] == 31.56
    assert by_currency["EUR"]["total"] == 100.00  # валюты не смешиваются
    largest = by_currency["RUB"]["largest_operations"]
    assert [op["description"] for op in largest] == ["Ресторан", "Пятёрочка", "Кофейня"]


def test_summary_by_month_and_income(tools: ToolRegistry) -> None:
    months = tools.execute("spending_summary", {"group_by": "month", "category": "Кафе и рестораны"})
    rows = months["result"]["by_currency"][0]["groups"]
    assert [(row["group"], row["total"]) for row in rows] == [("2026-09", 2000.00), ("2026-10", 300.00)]
    income = tools.execute("spending_summary", {"direction": "income"})
    assert income["result"]["by_currency"][0]["total"] == 50000.00


def test_search_filters_sorts_and_limits(tools: ToolRegistry) -> None:
    answer = tools.execute("search_transactions", {"direction": "expense", "sort": "amount_desc", "limit": 2})
    result = answer["result"]
    assert result["total_found"] == 8 and result["returned"] == 2 and result["truncated"]
    assert [op["amount"] for op in result["operations"]] == [-1749.50, -1000.00]
    by_text = tools.execute("search_transactions", {"text": "кофей"})["result"]
    assert by_text["total_found"] == 2


def test_recurring_payments_find_monthly_subscription(tools: ToolRegistry) -> None:
    result = tools.execute("find_recurring_payments", {})["result"]
    assert [p["description"] for p in result["payments"]] == ["Подписка Музыка"]
    payment = result["payments"][0]
    assert payment["frequency"] == "ежемесячно" and payment["next_expected_date"] == "2026-11-04"
    assert result["monthly_cost_by_currency"] == {"RUB": 169.00}


def test_invalid_arguments_return_readable_errors(tools: ToolRegistry) -> None:
    answer = tools.execute("search_transactions", {
        "date_from": "2026-09-31", "category": "еда", "limit": 500, "unknown": 1,
    })
    assert not answer["ok"]
    error = answer["error"]
    assert error["code"] == "invalid_arguments"
    text = " ".join(error["details"])
    assert "date_from" in text and "category" in text and "limit: должно быть не больше 50" in text
    assert "unknown: неизвестный аргумент" in text


def test_period_order_and_json_string_arguments(tools: ToolRegistry) -> None:
    reversed_period = tools.execute("spending_summary", {"date_from": "2026-10-01", "date_to": "2026-09-01"})
    assert "date_from позже date_to" in reversed_period["error"]["details"][0]
    from_json = tools.execute("search_transactions", '{"limit": 1}')
    assert from_json["ok"] and from_json["result"]["returned"] == 1
    broken = tools.execute("search_transactions", "{limit: 1")
    assert broken["error"]["code"] == "invalid_arguments"


def test_empty_result_is_success_with_note(tools: ToolRegistry) -> None:
    answer = tools.execute("search_transactions", {"date_from": "2025-01-01", "date_to": "2025-01-31"})
    assert answer["ok"]
    assert answer["result"]["total_found"] == 0
    assert "вне выписки" in answer["result"]["period"]["note"]


def test_unknown_tool_and_missing_statement() -> None:
    tools = make_tools(StatementStore())
    assert tools.execute("transfer_money", {})["error"]["code"] == "unknown_tool"
    assert tools.execute("spending_summary", {})["error"]["code"] == "no_statement"


def test_schemas_are_simple_and_match_arguments(tools: ToolRegistry) -> None:
    for spec in tools.specifications():
        params = spec["parameters"]
        assert params["type"] == "object"
        assert "anyOf" not in str(params)
    search = tools.specifications()[0]["parameters"]["properties"]
    assert search["limit"]["maximum"] == 50 and "кафе и рестораны" in search["category"]["enum"]


def test_text_search_ignores_alphabet_and_case(tools: ToolRegistry) -> None:
    assert tools.execute("search_transactions", {"text": "hotel"})["result"]["total_found"] == 1
    assert tools.execute("search_transactions", {"text": "ХОТЕЛ"})["result"]["total_found"] == 1
    assert tools.execute("search_transactions", {"text": "pyaterochka"})["result"]["total_found"] == 1
