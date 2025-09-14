from __future__ import annotations
import os, random, re, glob, argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import requests
from ai_text import generate_push_ai

FX_RATES: dict[str, float] = {
    "KZT": 1.0,
    "USD": 480.0,
    "EUR": 520.0,
    "RUB": 5.0,
}

PRODUCTS: list[str] = [
    "Карта для путешествий",
    "Премиальная карта",
    "Кредитная карта",
    "Обмен валют",
    "Депозит Сберегательный",
    "Депозит Накопительный",
    "Депозит Мультивалютный",
    "Инвестиции",
    "Золотые слитки",
    "Кредит наличными",
]

TRAVEL_CATEGORIES = {"Такси", "Путешествия", "Отели"}
PREMIUM_BOOST_CATEGORIES = {"Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"}
ONLINE_SERVICE_CATEGORIES = {"Смотрим дома", "Играем дома", "Кино"}

PREMIUM_CASHBACK_BASE = [
    (6_000_000, 0.04),  # >= 6 млн → 4%
    (1_000_000, 0.03),  # 1–6 млн → 3%
    (0, 0.02),          # иначе 2%
]
PREMIUM_CAP_PER_MONTH = 100_000  # лимит кешбэка/мес

RU_MONTHS_GEN = [
    "январе", "феврале", "марте", "апреле", "мае", "июне",
    "июле", "августе", "сентябре", "октябре", "ноябре", "декабре"
]


def month_genitive(dt: datetime) -> str:
    return RU_MONTHS_GEN[(dt or datetime.now()).month - 1]

def fmt_money_kzt(amount: float, prec: int = 0) -> str:
    if amount is None:
        amount = 0.0
    if prec == 0:
        s = f"{round(float(amount)):,}".replace(",", " ")
    else:
        s = f"{float(amount):,.{prec}f}".replace(",", " ").replace(".", ",")
    return f"{s} ₸"


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Папка не найдена: {data_dir}")

    clients_path = None
    for f in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, f)) and f.lower() == "clients.csv":
            clients_path = os.path.join(data_dir, f)
            break
    if clients_path is None:
        raise FileNotFoundError("Не найден clients.csv")

    clients = pd.read_csv(clients_path)

    def pick(patterns: list[str]) -> list[str]:
        out = []
        for p in patterns:
            out.extend(glob.glob(os.path.join(data_dir, p)))
        return sorted(set(out))

    tx_files = pick([
        "client_*_transactions_3m.csv",
        "client_*_transactions_3m",
        "client_*_transactions.csv",
        "client_*_transactions",
    ])
    tr_files = pick([
        "client_*_transfers_3m.csv",
        "client_*_transfers_3m",
        "client_*_transfers.csv",
        "client_*_transfers",
    ])

    if not tx_files:
        raise FileNotFoundError("Не найдено client_*_transactions_3m(.csv)")
    if not tr_files:
        raise FileNotFoundError("Не найдено client_*_transfers_3m(.csv)")

    def read_tx(path):
        df = pd.read_csv(path)
        need = ["client_code", "date", "category", "amount", "currency"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)}: нет колонок {missing}")
        keep = need + [c for c in ["name", "status", "age", "city"] if c in df.columns]
        return df[keep].copy()

    def read_tr(path):
        df = pd.read_csv(path)
        need = ["client_code", "date", "type", "direction", "amount", "currency"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)}: нет колонок {missing}")
        keep = need + [c for c in ["name", "status", "age", "city"] if c in df.columns]
        return df[keep].copy()

    tx = pd.concat([read_tx(p) for p in tx_files], ignore_index=True)
    tr = pd.concat([read_tr(p) for p in tr_files], ignore_index=True)

    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
    tr["date"] = pd.to_datetime(tr["date"], errors="coerce")

    return clients, tx, tr


def _kzt(amount, cur) -> float:
    try:
        return float(amount) * float(FX_RATES.get(str(cur), 1.0))
    except Exception:
        return 0.0

def compute_teacher_benefits(client_row: pd.Series,
                             tx_c: pd.DataFrame,
                             tr_c: pd.DataFrame) -> tuple[dict[str, float], dict]:
    avg_balance = float(client_row.get("avg_monthly_balance_KZT", 0.0) or 0.0)

    if not tx_c.empty:
        tx_c = tx_c.copy()
        tx_c["amount_kzt"] = [
            _kzt(a, cur) for a, cur in zip(pd.to_numeric(tx_c["amount"], errors="coerce").fillna(0.0), tx_c["currency"])
        ]
        spend_total = float(tx_c["amount_kzt"].sum())
        spend_by_cat = tx_c.groupby("category", dropna=False)["amount_kzt"].sum().sort_values(ascending=False)
        top3 = list(spend_by_cat.head(3).index.astype(str))
        travel_taxi_sum = float(tx_c[tx_c["category"].isin(TRAVEL_CATEGORIES)]["amount_kzt"].sum())
        last_dt = tx_c["date"].dropna().max()
    else:
        spend_total = 0.0
        spend_by_cat = pd.Series(dtype=float)
        top3 = []
        travel_taxi_sum = 0.0
        last_dt = pd.Timestamp(datetime.now())

    if not tr_c.empty:
        tr_c = tr_c.copy()
        tr_c["amount_kzt"] = [
            _kzt(a, cur) for a, cur in zip(pd.to_numeric(tr_c["amount"], errors="coerce").fillna(0.0), tr_c["currency"])
        ]
    else:
        tr_c = pd.DataFrame(columns=["type", "direction", "amount_kzt", "currency"])

    non_kzt_spend = 0.0
    fx_signal_by_curr: dict[str, float] = {}

    if not tx_c.empty:
        non_kzt_df = tx_c[tx_c["currency"].astype(str) != "KZT"].copy()
        if not non_kzt_df.empty:
            non_kzt_spend = float(non_kzt_df["amount_kzt"].sum())
            for cur in non_kzt_df["currency"].astype(str).unique():
                cur_sum = float(non_kzt_df[non_kzt_df["currency"] == cur]["amount_kzt"].sum())
                fx_signal_by_curr[cur] = fx_signal_by_curr.get(cur, 0.0) + cur_sum

    fx_tr = tr_c[tr_c["type"].isin(["fx_buy", "fx_sell"])].copy()
    fx_volume = float(fx_tr["amount_kzt"].sum()) if not fx_tr.empty else 0.0

    for cur in tr_c["currency"].astype(str).unique():
        if cur and cur != "KZT":
            cur_sum = float(tr_c[tr_c["currency"] == cur]["amount_kzt"].sum())
            fx_signal_by_curr[cur] = fx_signal_by_curr.get(cur, 0.0) + cur_sum

    fx_curr = max(fx_signal_by_curr.items(), key=lambda x: x[1])[0] if fx_signal_by_curr else None

    benefits: dict[str, float] = {p: 0.0 for p in PRODUCTS}

    benefits["Карта для путешествий"] = 0.04 * travel_taxi_sum

    def premium_rate(balance: float) -> float:
        for threshold, rate in PREMIUM_CASHBACK_BASE:
            if balance >= threshold:
                return rate
        return 0.02

    base_rate = premium_rate(avg_balance)
    boost_sum = float(tx_c[tx_c["category"].isin(PREMIUM_BOOST_CATEGORIES)]["amount_kzt"].sum()) if not tx_c.empty else 0.0
    premium_cashback = base_rate * spend_total + 0.04 * boost_sum

    if not tx_c.empty and tx_c["date"].notna().any():
        months_cnt = int(tx_c["date"].dt.to_period("M").nunique())
        months_cnt = max(1, min(3, months_cnt))
    else:
        months_cnt = 3
    cap = PREMIUM_CAP_PER_MONTH * months_cnt
    benefits["Премиальная карта"] = min(premium_cashback, cap)

    if not spend_by_cat.empty:
        top3_cats = list(spend_by_cat.head(3).index.astype(str))
        top3_sum = float(spend_by_cat.head(3).sum())
        online_sum_full = float(tx_c[tx_c["category"].isin(ONLINE_SERVICE_CATEGORIES)]["amount_kzt"].sum())
        online_extra = float(tx_c[
            tx_c["category"].isin(ONLINE_SERVICE_CATEGORIES - set(top3_cats))
        ]["amount_kzt"].sum())
        benefits["Кредитная карта"] = 0.10 * (top3_sum + online_extra)
    else:
        top3_cats = []
        benefits["Кредитная карта"] = 0.0

    benefits["Обмен валют"] = 0.005 * (fx_volume + non_kzt_spend)

    q = 3.0 / 12.0
    sber_share = 0.7
    acc_share = 0.5
    multi_share = 0.5 if (fx_volume > 0.0 or non_kzt_spend > 0.0) else 0.0

    benefits["Депозит Сберегательный"] = avg_balance * 0.165 * q * sber_share
    benefits["Депозит Накопительный"] = avg_balance * 0.155 * q * acc_share
    benefits["Депозит Мультивалютный"] = avg_balance * 0.145 * q * multi_share

    benefits["Инвестиции"] = avg_balance * 0.02 * q * 0.30  # ~0.5% за квартал от 30% остатка

    gold_ops = tr_c[tr_c["type"].isin(["gold_buy_out", "gold_sell_in"])]
    benefits["Золотые слитки"] = avg_balance * 0.001 if not gold_ops.empty else 0.0

    cash_signals = tr_c[tr_c["type"].isin([
        "atm_withdrawal", "loan_payment_out", "installment_payment_out", "cc_repayment_out"
    ])]
    signal_sum = float(cash_signals["amount_kzt"].sum()) if not cash_signals.empty else 0.0
    benefits["Кредит наличными"] = 0.005 * signal_sum if signal_sum >= 50_000 else 0.0

    name = str(client_row.get("name")) if pd.notna(client_row.get("name")) else f"Клиент {int(client_row['client_code'])}"
    now = last_dt.to_pydatetime() if pd.notna(last_dt) else datetime.now()
    context = {
        "name": name,
        "now": now,
        "top3": top3_cats if top3_cats else top3,
        "fx_curr": fx_curr,
        "travel_taxi_sum": travel_taxi_sum,
        "avg_balance": avg_balance,
    }
    return benefits, context

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="case 1", help="Папка с clients.csv и client_*_transactions_3m/transfers_3m")
    ap.add_argument("--out", default="output.csv", help="Итоговый CSV: client_code,product,push_notification")
    ap.add_argument("--debug", default="debug_top4.csv", help="Отладочный CSV с топ-4 продуктов")
    # ИИ
    ap.add_argument("--llm", choices=["openai", "ollama"], default="openai",
                    help="Где генерировать текст пуша: openai или ollama")
    ap.add_argument("--llm-model", default="gpt-4o-mini",
                    help="Имя модели (напр. gpt-4o-mini, llama3:8b-instruct, qwen2.5:7b-instruct)")
    ap.add_argument("--ollama-url", default="http://localhost:11434",
                    help="URL Ollama, если --llm=ollama")
    args = ap.parse_args()

    clients, tx, tr = load_data(args.data_dir)

    rows = []
    dbg_rows = []

    for _, row in clients.iterrows():
        code = int(row["client_code"])
        random.seed(code)

        tx_c = tx[tx["client_code"] == code]
        tr_c = tr[tr["client_code"] == code]

        benefits, ctx = compute_teacher_benefits(row, tx_c, tr_c)

        ordered = sorted(benefits.items(), key=lambda x: -x[1])
        best_product, best_benefit = ordered[0]

        push_text = generate_push_ai(
                product=best_product,
                ctx=ctx,
                benefit_kzt=best_benefit,
                llm=args.llm,
                model=args.llm_model,
                ollama_url=args.ollama_url,
                temperature=0.4,
                tries=2,
        )

        rows.append({
            "client_code": code,
            "product": best_product,
            "push_notification": push_text,
        })

        dbg_rows.append({
            "client_code": code,
            "top1": f"{ordered[0][0]}:{round(ordered[0][1])}",
            "top2": f"{ordered[1][0]}:{round(ordered[1][1])}" if len(ordered) > 1 else "",
            "top3": f"{ordered[2][0]}:{round(ordered[2][1])}" if len(ordered) > 2 else "",
            "top4": f"{ordered[3][0]}:{round(ordered[3][1])}" if len(ordered) > 3 else "",
        })

    out_df = pd.DataFrame(rows)
    dbg_df = pd.DataFrame(dbg_rows)

    out_df.to_csv(args.out, index=False, encoding="utf-8")
    dbg_df.to_csv(args.debug, index=False, encoding="utf-8")

    print(f"Wrote {args.out} and {args.debug}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
