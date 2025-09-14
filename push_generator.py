import os, re, glob, argparse
import numpy as np
import pandas as pd
from collections import defaultdict

FX_RATES = {"KZT": 1.0, "USD": 480.0, "EUR": 520.0, "RUB": 5.0}
TRAVEL_CATEGORIES = {"Путешествия", "Такси", "Отели"}
PREMIUM_BOOST_CATEGORIES = {"Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"}
ONLINE_SERVICE_CATEGORIES = {"Смотрим дома", "Играем дома", "Кино"}
FX_TYPES = {"fx_buy", "fx_sell", "deposit_fx_topup_out", "deposit_fx_withdraw_in"}
GOLD_HINT_TYPES = {"gold_buy_out", "gold_sell_in"}

PRODUCTS = [
    "Карта для путешествий","Премиальная карта","Кредитная карта","Обмен валют",
    "Депозит Мультивалютный","Депозит Сберегательный","Депозит Накопительный",
    "Инвестиции","Кредит наличными","Золотые слитки",
]

TEMPLATES = {
    "Карта для путешествий": "{name}, в {month_name} у вас много поездок и такси. С тревел-картой часть расходов вернулась бы кешбэком — ≈ {benefit}. Оформить карту.",
    "Премиальная карта": "{name}, у вас стабильный остаток и траты в ресторанах. Премиальная карта даст повышенный кешбэк и бесплатные снятия — ≈ {benefit}. Оформить сейчас.",
    "Кредитная карта": "{name}, ваши топ-категории — {cat1}, {cat2}, {cat3}. До 10% в любимых категориях и онлайн-сервисах — ≈ {benefit}. Оформить карту.",
    "Обмен валют": "{name}, вы часто платите в {fx_curr}. Выгодный обмен и авто-покупка по цели — выгода ≈ {benefit}. Настроить обмен.",
    "Депозит Мультивалютный": "{name}, остаются свободные средства и траты в валюте. Разместите часть на мультивалютном вкладе — ≈ {benefit} за 3 мес. Открыть вклад.",
    "Депозит Сберегательный": "{name}, свободные средства можно разместить под 16,5% — ≈ {benefit} за 3 мес. Открыть вклад.",
    "Депозит Накопительный": "{name}, удобно копить под 15,5% — ≈ {benefit} за 3 мес. Открыть вклад.",
    "Инвестиции": "{name}, попробуйте инвестиции с низким порогом и без комиссий на старт — ≈ {benefit}. Открыть счёт.",
    "Кредит наличными": "{name}, если нужен запас на крупные траты — можно оформить кредит наличными. Узнать лимит.",
    "Золотые слитки": "{name}, для долгосрочного сохранения — рассмотрите золотые слитки 999,9. Узнать подробности.",
}

MONTHS_RU = {1:"январе",2:"феврале",3:"марте",4:"апреле",5:"мае",6:"июне",7:"июле",8:"августе",9:"сентябре",10:"октябре",11:"ноябре",12:"декабре"}

def kzt(amount, currency): return float(amount) * float(FX_RATES.get(str(currency), 1.0))
def fmt_kzt(amount):
    v = max(0.0, float(amount)); v = round(v/100.0)*100.0
    return f"{v:,.0f}".replace(",", " ") + " ₸"
def month_name_from_last(dts: pd.Series):
    if len(dts)==0 or dts.isna().all(): return "последний месяц"
    last = pd.to_datetime(dts, errors="coerce").max()
    return MONTHS_RU.get(int(getattr(last,"month",0)), "последний месяц")
def top3_categories(spend_by_cat):
    items = sorted(spend_by_cat.items(), key=lambda x: x[1], reverse=True)
    cats = [c for c,v in items if v>0]
    while len(cats)<3: cats.append("повседневные покупки")
    return cats[:3]
def render_push(product, context, benefit_value):
    txt = TEMPLATES.get(product, "{name}, для вас есть продукт — выгода ≈ {benefit}. Посмотреть.").format(
        **{**context, "benefit": fmt_kzt(benefit_value)}
    )
    if len(txt)>220:
        parts = txt.split(". ")
        if len(parts)>2:
            core,cta = ". ".join(parts[:-1]), parts[-1]
            core = core[:200].rstrip(" ,;:")
            txt = f"{core}. {cta}"
        txt = txt[:220].rstrip(" .,!;:") + "."
    return txt

def read_many(data_dir, patterns):
    files=[]
    for p in patterns: files+=glob.glob(os.path.join(data_dir,p))
    return sorted(set(files))
def load_data(data_dir: str):
    # 0) Базовая проверка папки + отладка
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Папка не найдена: {data_dir}")
    all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    if not all_files:
        raise FileNotFoundError(f"В папке {data_dir} нет файлов. Проверь путь.")
    print(f"[INFO] data_dir={data_dir}, файлов: {len(all_files)} (первые 10): {all_files[:10]}")

    clients_path = None
    for f in all_files:
        if f.lower() == "clients.csv":
            clients_path = os.path.join(data_dir, f)
            break
    if clients_path is None:
        raise FileNotFoundError("Не найден clients.csv (регистр важен).")
    clients = pd.read_csv(clients_path)

    tx_candidates, tr_candidates = [], []
    for f in all_files:
        fl = f.lower()
        if ("transactions" in fl) and ("3m" in fl or "transactions" in fl):
            if re.search(r"client_.*transactions", fl):  # client_*_transactions...
                tx_candidates.append(os.path.join(data_dir, f))
        if "transfers" in fl and re.search(r"client_.*transfers", fl):
            tr_candidates.append(os.path.join(data_dir, f))

    if not tx_candidates:
        for f in all_files:
            if f.lower().startswith("transactions"):
                tx_candidates.append(os.path.join(data_dir, f))
                break
    if not tr_candidates:
        for f in all_files:
            if f.lower().startswith("transfers"):
                tr_candidates.append(os.path.join(data_dir, f))
                break

    if not tx_candidates or not tr_candidates:
        raise FileNotFoundError(
            "Не нашёл операции: нужны client_*_transactions_3m(.csv) и client_*_transfers(.csv) "
            "или общие transactions(.csv)/transfers(.csv)."
        )

    print(f"[INFO] Найдено transactions-файлов: {len(tx_candidates)}")
    print(f"[INFO] Найдено transfers-файлов: {len(tr_candidates)}")

    def read_tx(path):
        df = pd.read_csv(path)
        need = ["client_code","date","category","amount","currency"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)}: нет колонок {missing}")
        return df[need].copy()

    def read_tr(path):
        df = pd.read_csv(path)
        need = ["client_code","date","type","direction","amount","currency"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)}: нет колонок {missing}")
        return df[need].copy()

    tx = pd.concat([read_tx(p) for p in tx_candidates], ignore_index=True)
    tr = pd.concat([read_tr(p) for p in tr_candidates], ignore_index=True)

    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
    tr["date"] = pd.to_datetime(tr["date"], errors="coerce")

    print(f"[OK] transactions: {len(tx)} строк, transfers: {len(tr)} строк, clients: {len(clients)}")
    return clients, tx, tr

# Для обучения модели
def compute_teacher_benefits(client_row, tx_c, tr_c):
    avg_bal = float(client_row.get("avg_monthly_balance_KZT",0) or 0.0)

    spend_by_cat, total_spend, non_kzt_spend_kzt = {}, 0.0, 0.0
    if not tx_c.empty:
        tx = tx_c.copy()
        tx["amount_kzt"]=[kzt(a,c) for a,c in zip(tx["amount"], tx["currency"])]
        spend_by_cat = tx.groupby("category")["amount_kzt"].sum().to_dict()
        total_spend = float(tx["amount_kzt"].sum())
        non_kzt = tx[tx["currency"].ne("KZT")]
        if not non_kzt.empty:
            rate = non_kzt["currency"].map(FX_RATES).fillna(1.0).astype(float)
            amt = non_kzt["amount"].astype(float)
            non_kzt_spend_kzt = float((amt*rate).sum())
        month_name = month_name_from_last(tx["date"])
    else:
        month_name = "последний месяц"

    fx_volume_kzt, has_gold_hint = 0.0, False
    loan_signals=set()
    if not tr_c.empty:
        tr = tr_c.copy()
        tr["amount_kzt"]=[kzt(a,c) for a,c in zip(tr["amount"], tr["currency"])]
        fx_rows = tr[tr["type"].isin(FX_TYPES)]
        fx_volume_kzt = float(fx_rows["amount_kzt"].abs().sum())
        has_gold_hint = any(tr["type"].isin(GOLD_HINT_TYPES))
        loan_signals = set(tr["type"].unique())

    benefits = defaultdict(float)
    travel_spend = sum(spend_by_cat.get(cat,0.0) for cat in TRAVEL_CATEGORIES)
    benefits["Карта для путешествий"] = 0.04 * travel_spend
    base_rate = 0.04 if avg_bal>=6_000_000 else (0.03 if avg_bal>=1_000_000 else 0.02)
    boosted_sum = sum(spend_by_cat.get(cat,0.0) for cat in PREMIUM_BOOST_CATEGORIES)
    premium_cashback = base_rate*total_spend + max(0.0,0.04-base_rate)*boosted_sum
    premium_cashback = min(premium_cashback, 100_000.0*3)
    benefits["Премиальная карта"] = premium_cashback
    cats_sorted = sorted(spend_by_cat.items(), key=lambda x:x[1], reverse=True)
    top3_sum = sum(v for _,v in cats_sorted[:3])
    online_sum = sum(spend_by_cat.get(cat,0.0) for cat in ONLINE_SERVICE_CATEGORIES)
    benefits["Кредитная карта"] = 0.10*(top3_sum+online_sum)
    benefits["Обмен валют"] = 0.005*(fx_volume_kzt + non_kzt_spend_kzt)
    benefits["Депозит Сберегательный"] = (0.70 if avg_bal>0 else 0.0)*avg_bal*0.165*(3/12)
    benefits["Депозит Накопительный"]  = (0.50 if avg_bal>0 else 0.0)*avg_bal*0.155*(3/12)
    benefits["Депозит Мультивалютный"] = (0.50 if (avg_bal>0 and (fx_volume_kzt>0 or non_kzt_spend_kzt>0)) else 0.0)*avg_bal*0.145*(3/12)
    benefits["Инвестиции"] = (0.30 if avg_bal>0 else 0.0)*avg_bal*0.02*(3/12)
    benefits["Золотые слитки"] = 0.001*avg_bal if has_gold_hint else 0.0
    needs_cash = any(t in loan_signals for t in {"atm_withdrawal","installment_payment_out","loan_payment_out"})
    benefits["Кредит наличными"] = 5_000.0 if needs_cash else 0.0

    cats3 = top3_categories(spend_by_cat)
    fx_curr = "валюте"
    if not tx_c.empty:
        nonk = tx_c[tx_c["currency"].ne("KZT")]["currency"].value_counts()
        if len(nonk)>0: fx_curr=nonk.index[0]
    fx_curr = {"USD":"долларах","EUR":"евро","RUB":"рублях"}.get(fx_curr,"валюте")

    ctx = {"name": client_row["name"], "month_name": month_name,
           "cat1": cats3[0], "cat2": cats3[1], "cat3": cats3[2],
           "fx_curr": fx_curr}
    return dict(benefits), ctx

def engineer_features(client_row, tx_c, tr_c):
    avg_bal = float(client_row.get("avg_monthly_balance_KZT",0) or 0.0)
    spend_by_cat, total_spend, non_kzt_spend_kzt, months_active = {}, 0.0, 0.0, 0
    if not tx_c.empty:
        tx = tx_c.copy()
        tx["amount_kzt"]=[kzt(a,c) for a,c in zip(tx["amount"],tx["currency"])]
        spend_by_cat = tx.groupby("category")["amount_kzt"].sum().to_dict()
        total_spend = float(tx["amount_kzt"].sum())
        non_kzt = tx[tx["currency"].ne("KZT")]
        if not non_kzt.empty:
            rate = non_kzt["currency"].map(FX_RATES).fillna(1.0).astype(float)
            amt = non_kzt["amount"].astype(float)
            non_kzt_spend_kzt = float((amt*rate).sum())
        months_active = int(tx["date"].dt.to_period("M").nunique())
    travel_sum = sum(spend_by_cat.get(c,0.0) for c in TRAVEL_CATEGORIES)
    premium_boost_sum = sum(spend_by_cat.get(c,0.0) for c in PREMIUM_BOOST_CATEGORIES)
    online_sum = sum(spend_by_cat.get(c,0.0) for c in ONLINE_SERVICE_CATEGORIES)

    fx_volume_kzt, atm_cnt, loan_cnt, invest_cnt, gold_cnt = 0.0,0,0,0,0
    if not tr_c.empty:
        tr = tr_c.copy()
        tr["amount_kzt"]=[kzt(a,c) for a,c in zip(tr["amount"],tr["currency"])]
        fx_volume_kzt = float(tr[tr["type"].isin(FX_TYPES)]["amount_kzt"].abs().sum())
        atm_cnt   = int((tr["type"]=="atm_withdrawal").sum())
        loan_cnt  = int(tr["type"].isin({"installment_payment_out","loan_payment_out"}).sum())
        invest_cnt= int(tr["type"].isin({"invest_in","invest_out"}).sum())
        gold_cnt  = int(tr["type"].isin({"gold_buy_out","gold_sell_in"}).sum())

    status = str(client_row.get("status",""))
    status_vec = [
        1.0 if status=="Студент" else 0.0,
        1.0 if status=="Зарплатный клиент" else 0.0,
        1.0 if status=="Премиальный клиент" else 0.0,
        1.0 if status=="Стандартный клиент" else 0.0,
    ]

    feats = [
        avg_bal, total_spend, travel_sum, premium_boost_sum, online_sum,
        non_kzt_spend_kzt, fx_volume_kzt, months_active,
        atm_cnt, loan_cnt, invest_cnt, gold_cnt,
    ] + status_vec

    for i in range(0,7): feats[i]=float(np.log1p(max(feats[i],0.0)))
    return np.array(feats, dtype=np.float64)

def ridge_fit(Z, Y, lam=1e-3):
    Zb = np.hstack([Z, np.ones((Z.shape[0],1))])
    I  = np.eye(Zb.shape[1]); I[-1,-1]=0.0
    W  = np.linalg.solve(Zb.T@Zb + lam*I, Zb.T@Y)
    return W
def standardize(X, mu=None, sigma=None):
    mu = X.mean(axis=0) if mu is None else mu
    sigma = X.std(axis=0) if sigma is None else sigma
    sigma[sigma==0]=1.0
    Z = (X-mu)/sigma
    return Z, mu, sigma
def top1_accuracy(teacher_top1, student_top1):
    return sum(int(a==b) for a,b in zip(teacher_top1, student_top1))/len(teacher_top1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="case 1")
    ap.add_argument("--out", default="output.csv")
    ap.add_argument("--debug", default="debug_top4.csv")
    ap.add_argument("--metrics", default="metrics.txt")
    ap.add_argument("--mode", choices=["push", "contracts"], default="push")
    ap.add_argument("--contracts-input", default=None)
    ap.add_argument("--contracts-out", default="contracts.json")
    args = ap.parse_args()

    clients, tx, tr = load_data(args.data_dir)

    X_list, Y_list, ctxs, ids, teacher_top1 = [], [], [], [], []
    for _, row in clients.iterrows():
        code = int(row["client_code"])
        tx_c = tx[tx["client_code"]==code]
        tr_c = tr[tr["client_code"]==code]
        benefits, ctx = compute_teacher_benefits(row, tx_c, tr_c)
        y = [benefits.get(p,0.0) for p in PRODUCTS]
        top = max(benefits.items(), key=lambda x:x[1])[0] if benefits else PRODUCTS[0]
        feats = engineer_features(row, tx_c, tr_c)

        X_list.append(feats); Y_list.append(y); ctxs.append(ctx); ids.append(code); teacher_top1.append(top)

    X = np.stack(X_list); Y = np.array(Y_list, dtype=np.float64)

    rng = np.random.default_rng(42)
    idx = np.arange(len(ids)); rng.shuffle(idx)
    cut = int(0.8*len(ids)); tr_idx, va_idx = idx[:cut], idx[cut:]

    Z_tr, mu, sigma = standardize(X[tr_idx])
    W = ridge_fit(Z_tr, Y[tr_idx], lam=1e-3)

    Z_all, _, _ = standardize(X, mu, sigma)
    Y_pred = np.hstack([Z_all, np.ones((Z_all.shape[0],1))]) @ W
    Y_pred = np.maximum(0.0, Y_pred)

    rows, dbg = [], []
    student_top1 = []
    for i, code in enumerate(ids):
        order = np.argsort(-Y_pred[i])
        best_product = PRODUCTS[order[0]]
        student_top1.append(best_product)
        push = render_push(best_product, ctxs[i], float(Y_pred[i, order[0]]))
        rows.append({"client_code": code, "product": best_product, "push_notification": push})
        dbg.append({
            "client_code": code,
            "student_top1": PRODUCTS[order[0]],
            "student_top2": PRODUCTS[order[1]] if len(order)>1 else "",
            "student_top3": PRODUCTS[order[2]] if len(order)>2 else "",
            "student_top4": PRODUCTS[order[3]] if len(order)>3 else "",
            "teacher_top1": teacher_top1[i],
            "predicted_benefit_best": round(float(Y_pred[i, order[0]]),2)
        })

    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8")
    pd.DataFrame(dbg).to_csv(args.debug, index=False, encoding="utf-8")

    acc = top1_accuracy(teacher_top1, student_top1)
    with open(args.metrics,"w",encoding="utf-8") as f:
        f.write(f"Top1(Student vs Teacher): {acc:.2%}\n")
    print(f"Wrote {args.out}, {args.debug}, {args.metrics}")

if __name__ == "__main__":
    main()
