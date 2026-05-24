from pathlib import Path

import pandas as pd


OUT = Path(__file__).parent


def sales_demo(rows: int = 5000) -> None:
    products = ["Laptop", "Mouse", "Monitor", "Keyboard", "Dock", "Camera"]
    regions = ["Irkutsk", "Moscow", "Kazan", "Novosibirsk"]
    data = []
    for i in range(rows):
        product = products[i % len(products)]
        quantity = (i % 9) + 1
        price = {"Laptop": 100000, "Mouse": 1500, "Monitor": 25000, "Keyboard": 7000, "Dock": 12000, "Camera": 9000}[product]
        date = "10.01.2025" if i % 3 == 1 else ("2025/01/11" if i % 3 == 2 else "2025-01-10")
        data.append(
            {
                "order_id": i + 1,
                "product": product,
                "region": regions[i % len(regions)],
                "price": price,
                "quantity": quantity,
                "discount": 0.05 if i % 10 == 0 else 0,
                "date": date,
                "comment": "" if i % 17 else "manual check",
            }
        )
    source = pd.DataFrame(data)
    source.to_csv(OUT / "large_sales_source.csv", index=False)
    expected = source.head(6).copy()
    expected["total"] = (expected["price"] * expected["quantity"] * (1 - expected["discount"])).round(2)
    expected["date"] = pd.to_datetime(expected["date"], dayfirst=True, errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    expected = expected[["order_id", "product", "region", "total", "date"]]
    expected.to_csv(OUT / "large_sales_expected.csv", index=False)


def customers_demo(rows: int = 3000) -> None:
    data = []
    for i in range(rows):
        data.append(
            {
                "client_id": f"C{i:05d}",
                "first_name": ["Anna", "Ivan", "Maria", "Petr"][i % 4],
                "last_name": ["Smirnova", "Petrov", "Ivanova", "Sokolov"][i % 4],
                "email": None if i % 23 == 0 else f"user{i}@example.com",
                "age": 120 if i % 997 == 0 else 18 + (i % 55),
                "signup_date": "01.02.2025" if i % 2 else "2025-02-01",
            }
        )
    source = pd.DataFrame(data)
    source.to_csv(OUT / "customers_source.csv", index=False)
    expected = source.head(5).copy()
    expected["full_name"] = expected["first_name"] + " " + expected["last_name"]
    expected["signup_date"] = pd.to_datetime(expected["signup_date"], dayfirst=True, format="mixed").dt.strftime("%Y-%m-%d")
    expected = expected[["client_id", "full_name", "email", "age", "signup_date"]]
    expected.to_csv(OUT / "customers_expected.csv", index=False)


if __name__ == "__main__":
    sales_demo()
    customers_demo()
    print("Demo tables generated in examples/")
