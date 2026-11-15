"""Генерация синтетической выписки: python scripts/generate_statement.py

Все данные вымышлены. Генерация детерминирована (фиксированный seed),
поэтому ожидаемые ответы для проверок не меняются между запусками.
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

OUTPUT: Path = Path(__file__).resolve().parent.parent / "data" / "sample_statement.csv"
START: date = date(2026, 8, 1)
END: date = date(2026, 10, 31)

# (описание, MCC, мин. сумма, макс. сумма, средне операций в неделю)
EVERYDAY: list[tuple[str, str, int, int, float]] = [
    ("Пятёрочка", "5411", 350, 2400, 2.0),
    ("Перекрёсток", "5411", 600, 3800, 1.0),
    ("ВкусВилл", "5499", 250, 1500, 1.0),
    ("Кофейня «Зерно»", "5814", 180, 450, 2.0),
    ("Ресторан «Сыроварня»", "5812", 1800, 5200, 0.4),
    ("Яндекс Go такси", "4121", 280, 1300, 1.2),
    ("Метро, оплата проезда", "4111", 63, 63, 4.0),
    ("Аптека «Ригла»", "5912", 300, 2200, 0.4),
    ("Ozon", "5399", 500, 6500, 0.7),
    ("Wildberries", "5311", 400, 5000, 0.6),
    ("Кинотеатр «Октябрь»", "7832", 450, 1400, 0.25),
    ("Перевод по СБП", "4829", 500, 5000, 0.5),
]

# (описание, MCC, сумма, день месяца)
MONTHLY: list[tuple[str, str, str, int]] = [
    ("Яндекс Плюс подписка", "5815", "-399.00", 5),
    ("Онлайн-кинотеатр Иви", "5815", "-399.00", 12),
    ("Облачное хранилище", "5818", "-149.00", 20),
    ("Фитнес-клуб «Пульс», абонемент", "7997", "-3500.00", 3),
    ("МТС, мобильная связь", "4814", "-650.00", 8),
    ("Домашний интернет", "4899", "-750.00", 10),
    ("ЖКУ, оплата квитанции", "4900", "-6400.00", 15),
]


def main() -> None:
    rnd = random.Random(42)
    rows: list[tuple[date, str, str, str, str]] = []

    day = START
    while day <= END:
        for description, mcc, low, high, per_week in EVERYDAY:
            if rnd.random() < per_week / 7:
                amount = low if low == high else rnd.randint(low, high)
                cents = "00" if low == high else f"{rnd.randint(0, 99):02d}"
                rows.append((day, description, f"-{amount}.{cents}", "RUB", mcc))
        day += timedelta(days=1)

    month = START
    while month <= END:
        for description, mcc, amount, day_of_month in MONTHLY:
            rows.append((month.replace(day=day_of_month), description, amount, "RUB", mcc))
        rows.append((month.replace(day=5), "Зарплата, аванс", "45000.00", "RUB", "6012"))
        rows.append((month.replace(day=20), "Зарплата", "65000.00", "RUB", "6012"))
        month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)

    # Разовые операции: валютные траты, кешбэк, снятие наличных и двойное списание.
    rows.append((date(2026, 8, 17), "Steam purchase", "-15.99", "USD", "5816"))
    rows.append((date(2026, 9, 9), "Hotel Lisboa Centro", "-240.00", "EUR", "7011"))
    rows.append((date(2026, 9, 9), "Lisbon Metro", "-12.50", "EUR", "4111"))
    rows.append((date(2026, 9, 1), "Кешбэк за август", "1240.55", "RUB", ""))
    rows.append((date(2026, 10, 2), "Снятие наличных, банкомат", "-5000.00", "RUB", "6011"))
    rows.append((date(2026, 10, 14), "Ozon", "-3290.00", "RUB", "5399"))
    rows.append((date(2026, 10, 14), "Ozon", "-3290.00", "RUB", "5399"))

    rows.sort(key=lambda row: row[0])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["operation_id", "date", "description", "amount", "currency", "mcc"])
        for number, (day, description, amount, currency, mcc) in enumerate(rows, start=1):
            writer.writerow([f"OP-{number:05d}", day.isoformat(), description, amount, currency, mcc])
    print(f"Записано {len(rows)} операций в {OUTPUT}")


if __name__ == "__main__":
    main()
