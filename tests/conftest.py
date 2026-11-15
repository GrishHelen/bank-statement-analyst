import pytest

from src.importers.csv_statement import Statement, parse_statement
from src.tools.base import ToolRegistry
from src.tools.statement_tools import StatementStore, make_tools

# Маленькая выписка с заранее посчитанными ответами:
# расходы сентября в RUB — 3169.00 (кафе 2000.00, продукты 1000.00, подписки 169.00), в EUR — 100.00.
SMALL_CSV: str = """operation_id,date,description,amount,currency,mcc
1,2026-09-03,Кофейня,-250.50,RUB,5814
2,2026-09-10,Ресторан,-1749.50,RUB,5812
3,2026-09-15,Пятёрочка,-1000.00,RUB,5411
4,2026-09-20,Зарплата,50000.00,RUB,6012
5,2026-10-05,Подписка Музыка,-169.00,RUB,5815
6,2026-09-05,Подписка Музыка,-169.00,RUB,5815
7,2026-08-05,Подписка Музыка,-169.00,RUB,5815
8,2026-09-09,Hotel,-100.00,EUR,7011
9,2026-10-01,Кофейня,-300.00,RUB,5814
"""


@pytest.fixture
def statement() -> Statement:
    return parse_statement(SMALL_CSV, source="test.csv")


@pytest.fixture
def store(statement: Statement) -> StatementStore:
    return StatementStore(statement)


@pytest.fixture
def tools(store: StatementStore) -> ToolRegistry:
    return make_tools(store)
