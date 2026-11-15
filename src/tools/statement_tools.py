"""Инструменты агента для работы с загруженной выпиской.

Все три инструмента только читают и считают: они не меняют данные и не
выполняют действий с последствиями. Суммы считаются в Decimal, разные
валюты никогда не складываются между собой.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.importers.csv_statement import Statement, Transaction
from src.importers.mcc import CATEGORIES
from src.tools.base import Tool, ToolError, ToolRegistry

Category = Literal[CATEGORIES]  # type: ignore[valid-type]
MAX_OPERATIONS: int = 50

# Транслитерация для поиска: «Озон» находит «Ozon», «яндекс» — «Yandex».
TRANSLIT: dict[int, str] = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya", "x": "ks", "w": "v",
})


def fold(text: str) -> str:
    """Текст для сравнения: нижний регистр, латиница, без лишних пробелов."""
    return " ".join(text.lower().translate(TRANSLIT).split())


class StatementStore:
    """Текущая выписка. Загружает её пользователь, а не модель."""

    def __init__(self, statement: Statement | None = None) -> None:
        self.statement = statement

    def require(self) -> Statement:
        if self.statement is None:
            raise ToolError(
                "no_statement",
                "Выписка не загружена.",
                ["Попросите пользователя загрузить CSV командой /load <путь к файлу>."],
            )
        return self.statement


# ---------- общие части схем ----------

class PeriodArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date_from: Optional[date] = Field(
        None, description="Начало периода включительно, ГГГГ-ММ-ДД. Не указывать — с начала выписки.")
    date_to: Optional[date] = Field(
        None, description="Конец периода включительно, ГГГГ-ММ-ДД. Не указывать — до конца выписки.")

    @field_validator("*", mode="before")
    @classmethod
    def empty_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("category", mode="before", check_fields=False)
    @classmethod
    def normalize_category(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @model_validator(mode="after")
    def check_period(self) -> "PeriodArgs":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from позже date_to")
        return self


class SearchArgs(PeriodArgs):
    category: Optional[Category] = Field(None, description="Категория операции из списка.")
    text: Optional[str] = Field(
        None, max_length=50,
        description="Подстрока в описании операции, без учёта регистра и алфавита: «озон» найдёт «Ozon».")
    direction: Literal["expense", "income", "all"] = Field(
        "all", description="expense — только расходы, income — только поступления, all — все.")
    min_amount: Optional[float] = Field(None, ge=0, le=1e9, description="Минимальная сумма по модулю.")
    max_amount: Optional[float] = Field(None, ge=0, le=1e9, description="Максимальная сумма по модулю.")
    sort: Literal["date_desc", "date_asc", "amount_desc"] = Field(
        "date_desc", description="Порядок: новые сначала, старые сначала или крупные по модулю сначала.")
    limit: int = Field(20, ge=1, le=MAX_OPERATIONS, description=f"Сколько операций вернуть, от 1 до {MAX_OPERATIONS}.")

    @model_validator(mode="after")
    def check_amounts(self) -> "SearchArgs":
        if self.min_amount is not None and self.max_amount is not None and self.min_amount > self.max_amount:
            raise ValueError("min_amount больше max_amount")
        return self


class SummaryArgs(PeriodArgs):
    direction: Literal["expense", "income"] = Field(
        "expense", description="expense — расходы, income — поступления.")
    group_by: Literal["category", "month", "description"] = Field(
        "category",
        description="Группировка: category — по категории, month — по месяцу, description — сумма по каждому "
                    "продавцу за период (это не отдельные покупки).")
    category: Optional[Category] = Field(None, description="Посчитать только эту категорию.")
    top_n: int = Field(10, ge=1, le=30, description="Сколько крупнейших групп показать, от 1 до 30.")


class RecurringArgs(PeriodArgs):
    min_occurrences: int = Field(
        3, ge=2, le=12, description="Минимум повторений, чтобы платёж считался регулярным, от 2 до 12.")
    amount_tolerance_pct: float = Field(
        10, ge=0, le=50, description="Допустимое отклонение суммы от типичной, в процентах, от 0 до 50.")


# ---------- общие функции ----------

def money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def select(statement: Statement, args: PeriodArgs) -> tuple[list[Transaction], dict[str, Any]]:
    """Операции за период и описание фактически использованного периода."""
    start = args.date_from or statement.date_from
    end = args.date_to or statement.date_to
    chosen = [t for t in statement.transactions if start <= t.date <= end]
    period: dict[str, Any] = {"date_from": start.isoformat(), "date_to": end.isoformat()}
    if end < statement.date_from or start > statement.date_to:
        period["note"] = (
            f"Запрошенный период вне выписки: она охватывает "
            f"{statement.date_from.isoformat()} — {statement.date_to.isoformat()}."
        )
    return chosen, period


def as_row(t: Transaction) -> dict[str, Any]:
    return {
        "operation_id": t.operation_id,
        "date": t.date.isoformat(),
        "description": t.description,
        "amount": money(t.amount),
        "currency": t.currency,
        "category": t.category,
    }


# ---------- инструменты ----------

def make_tools(store: StatementStore) -> ToolRegistry:
    def search_transactions(args: SearchArgs) -> dict[str, Any]:
        operations, period = select(store.require(), args)
        found: list[Transaction] = []
        for t in operations:
            if args.category and t.category != args.category:
                continue
            if args.text and fold(args.text) not in fold(t.description):
                continue
            if args.direction == "expense" and not t.is_expense:
                continue
            if args.direction == "income" and t.is_expense:
                continue
            if args.min_amount is not None and abs(t.amount) < Decimal(str(args.min_amount)):
                continue
            if args.max_amount is not None and abs(t.amount) > Decimal(str(args.max_amount)):
                continue
            found.append(t)
        if args.sort == "date_desc":
            found.sort(key=lambda t: (t.date, t.operation_id), reverse=True)
        elif args.sort == "amount_desc":
            found.sort(key=lambda t: abs(t.amount), reverse=True)
        returned = found[:args.limit]
        result: dict[str, Any] = {
            "period": period,
            "total_found": len(found),
            "returned": len(returned),
            "truncated": len(found) > len(returned),
            "operations": [as_row(t) for t in returned],
        }
        if not found:
            result["note"] = (
                "Операций по этим условиям нет. Это не ошибка: можно ослабить фильтры "
                "или расширить период."
            )
        return result

    def spending_summary(args: SummaryArgs) -> dict[str, Any]:
        operations, period = select(store.require(), args)
        groups: dict[str, dict[str, list[Decimal]]] = defaultdict(lambda: defaultdict(list))
        largest: dict[str, list[Transaction]] = defaultdict(list)
        for t in operations:
            if (args.direction == "expense") != t.is_expense:
                continue
            if args.category and t.category != args.category:
                continue
            if args.group_by == "category":
                key = t.category
            elif args.group_by == "month":
                key = t.date.strftime("%Y-%m")
            else:
                key = t.description
            groups[t.currency][key].append(abs(t.amount))
            largest[t.currency].append(t)

        by_currency: list[dict[str, Any]] = []
        for currency in sorted(groups):
            total = sum((sum(values, Decimal(0)) for values in groups[currency].values()), Decimal(0))
            rows = []
            for key, values in groups[currency].items():
                group_total = sum(values, Decimal(0))
                rows.append({
                    "group": key,
                    "total": money(group_total),
                    "count": len(values),
                    "share_pct": money(group_total * 100 / total) if total else 0.0,
                })
            if args.group_by == "month":
                rows.sort(key=lambda row: row["group"])
            else:
                rows.sort(key=lambda row: row["total"], reverse=True)
            shown = rows[:args.top_n]
            entry: dict[str, Any] = {
                "currency": currency,
                "total": money(total),
                "operations": sum(row["count"] for row in rows),
                "groups": shown,
                "largest_operations": [
                    as_row(t) for t in sorted(largest[currency], key=lambda t: abs(t.amount), reverse=True)[:3]
                ],
            }
            if len(rows) > len(shown):
                hidden = rows[len(shown):]
                entry["other_groups"] = {"count": len(hidden), "total": money(
                    sum((Decimal(str(row["total"])) for row in hidden), Decimal(0)))}
            by_currency.append(entry)

        result: dict[str, Any] = {
            "period": period,
            "direction": args.direction,
            "group_by": args.group_by,
            "by_currency": by_currency,
            "amounts_are": "суммы по модулю; разные валюты не складываются",
        }
        if not by_currency:
            result["note"] = "За этот период нет подходящих операций."
        return result

    def find_recurring_payments(args: RecurringArgs) -> dict[str, Any]:
        operations, period = select(store.require(), args)
        series: dict[tuple[str, str], list[Transaction]] = defaultdict(list)
        for t in operations:
            if t.is_expense:
                series[(t.description.lower(), t.currency)].append(t)

        tolerance = Decimal(str(args.amount_tolerance_pct)) / 100
        payments: list[dict[str, Any]] = []
        monthly_totals: dict[str, Decimal] = defaultdict(Decimal)
        for (_, currency), items in series.items():
            if len(items) < args.min_occurrences:
                continue
            items.sort(key=lambda t: t.date)
            amounts = [abs(t.amount) for t in items]
            typical = Decimal(str(statistics.median(amounts)))
            if any(abs(amount - typical) > typical * tolerance for amount in amounts):
                continue
            gaps = [(b.date - a.date).days for a, b in zip(items, items[1:])]
            median_gap = statistics.median(gaps)
            if 25 <= median_gap <= 35 and min(gaps) >= 20:
                frequency, per_month = "ежемесячно", Decimal(1)
            elif 6 <= median_gap <= 8 and min(gaps) >= 5:
                frequency, per_month = "еженедельно", Decimal("4.33")
            else:
                continue
            monthly = typical * per_month
            monthly_totals[currency] += monthly
            payments.append({
                "description": items[-1].description,
                "category": items[-1].category,
                "currency": currency,
                "typical_amount": money(typical),
                "frequency": frequency,
                "occurrences": len(items),
                "first_date": items[0].date.isoformat(),
                "last_date": items[-1].date.isoformat(),
                "next_expected_date": (items[-1].date + timedelta(days=round(median_gap))).isoformat(),
                "monthly_cost": money(monthly),
            })
        payments.sort(key=lambda p: p["monthly_cost"], reverse=True)
        result: dict[str, Any] = {
            "period": period,
            "rule": (
                f"расход с одинаковым описанием, не меньше {args.min_occurrences} раз, сумма отличается "
                f"от типичной не больше чем на {args.amount_tolerance_pct:g}%, интервал ~месяц или ~неделя"
            ),
            "payments": payments,
            "monthly_cost_by_currency": {c: money(v) for c, v in sorted(monthly_totals.items())},
        }
        if not payments:
            result["note"] = "Регулярных платежей по этому правилу не найдено."
        return result

    return ToolRegistry([
        Tool(
            name="search_transactions",
            kind="read",
            description=(
                "Найти отдельные операции в выписке по периоду, категории, тексту описания, "
                "направлению и сумме. Для «самых крупных покупок» используй direction=expense, "
                "sort=amount_desc и нужный limit. Возвращает до 50 операций и общее число найденных. "
                "Для итогов и сумм по категориям используй spending_summary, а не сложение вручную. "
                "Категории: " + ", ".join(CATEGORIES) + "."
            ),
            args_model=SearchArgs,
            function=search_transactions,
            examples=[
                ("Покажи 5 самых крупных трат в сентябре",
                 {"date_from": "2026-09-01", "date_to": "2026-09-30", "direction": "expense",
                  "sort": "amount_desc", "limit": 5}),
            ],
        ),
        Tool(
            name="spending_summary",
            kind="compute",
            description=(
                "Посчитать сумму расходов или поступлений за период с группировкой по категории, "
                "месяцу или описанию. Возвращает итог, число операций и долю каждой группы "
                "отдельно для каждой валюты, а также 3 крупнейшие отдельные операции в каждой валюте "
                "(largest_operations). Используй для любых вопросов «сколько потратил»."
            ),
            args_model=SummaryArgs,
            function=spending_summary,
            examples=[
                ("Сколько я потратил на кафе в октябре?",
                 {"date_from": "2026-10-01", "date_to": "2026-10-31", "direction": "expense",
                  "category": "кафе и рестораны", "group_by": "category"}),
            ],
        ),
        Tool(
            name="find_recurring_payments",
            kind="compute",
            description=(
                "Найти регулярные платежи и подписки: повторяющиеся расходы с одинаковым описанием, "
                "близкой суммой и интервалом около месяца или недели. Возвращает список платежей "
                "и их ежемесячную стоимость."
            ),
            args_model=RecurringArgs,
            function=find_recurring_payments,
            examples=[("Какие у меня есть подписки и сколько они стоят в месяц?", {})],
        ),
    ])
