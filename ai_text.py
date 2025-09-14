from __future__ import annotations
import argparse
import os
import glob
import random
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import requests

RU_MONTHS_GEN = [
    "январе","феврале","марте","апреле","мае","июне",
    "июле","августе","сентябре","октябре","ноябре","декабре"
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

def clamp_policies(text: str, max_len: int = 220) -> str:
    # максимум один "!"
    if text.count("!") > 1:
        parts = text.split("!")
        text = parts[0] + "!" + " ".join(p.strip() for p in parts[1:] if p.strip() and "." in p)
    # без тотального КАПС (оставим 2–3 буквенные аббревиатуры)
    def _decap(w: str) -> str:
        return w if (len(w) <= 3 and w.isupper()) else (w.capitalize() if w.isupper() else w)
    text = " ".join(_decap(w) for w in text.split())
    # мягкий трим
    if len(text) > max_len:
        text = text.replace(" — ", " ").replace("  ", " ")
    if len(text) > max_len:
        text = text[:max_len].rstrip(" ,.;:") + "."
    return text

def build_system_prompt() -> str:
    return (
        "Вы — помощник банка. Пишите ПУШ-уведомления на русском:\n"
        "- тон: на равных, доброжелательно, без воды и морали; обращение на «вы».\n"
        "- 1 мысль, 1 CTA.\n"
        "- длина 180–220 символов.\n"
        "- без КАПС; максимум один «!», не злоупотреблять им.\n"
        "- числа: пробел — разряд, запятая — дробная часть; валюта: «2 490 ₸».\n"
        "- структура: Контекст (по наблюдениям) → Польза (выгода) → CTA (глагол: Открыть/Оформить/Настроить/Посмотреть).\n"
        "Верните ответ ТОЛЬКО как JSON: {\"text\": \"...\"} без лишних полей."
    )

def build_user_prompt(product: str, ctx: Dict[str, any], benefit_kzt: float) -> str:
    name = ctx.get("name") or "Клиент"
    top3 = ctx.get("top3") or []
    top3_txt = ", ".join([str(x) for x in top3[:3]]) if top3 else ""
    fx_curr = ctx.get("fx_curr")
    avg_balance = float(ctx.get("avg_balance") or 0.0)
    travel_sum = float(ctx.get("travel_taxi_sum") or 0.0)
    month = month_genitive(ctx.get("now") or datetime.now())
    benefit_fmt = f"≈ {fmt_money_kzt(benefit_kzt, 0)}"

    hints = []
    if product == "Карта для путешествий":
        hints.append("упомяните поездки/такси/отели и что часть расходов вернётся кешбэком")
    elif product == "Премиальная карта":
        hints.append("упомяните высокий остаток/депозит и бесплатные снятия/повышенный кешбэк")
    elif product == "Кредитная карта":
        if top3_txt:
            hints.append(f"укажите топ-категории: {top3_txt}")
        hints.append("упомяните до 10% в любимых категориях и онлайн-сервисах")
    elif product == "Обмен валют":
        if fx_curr:
            hints.append(f"упомяните частые платежи в {fx_curr}")
        hints.append("упомяните выгодный обмен и авто-покупку по целевому курсу")
    elif product.startswith("Депозит"):
        hints.append("упомяните, что удобно копить и получать вознаграждение")
    elif product == "Инвестиции":
        hints.append("упомяните низкий порог входа и отсутствие комиссий на старт")
    elif product == "Кредит наличными":
        hints.append("упомяните гибкие выплаты и оформление без залога/справок")

    data = {
        "product": product,
        "name": name,
        "month_genitive": month,
        "benefit_formatted": benefit_fmt,
        "top3": top3[:3],
        "fx_curr": fx_curr,
        "avg_balance_kzt": avg_balance,
        "travel_taxi_sum_kzt": travel_sum,
        "write_rules": {
            "length": "180-220 chars",
            "one_exclamation_max": True,
            "no_caps": True,
            "one_thought_one_cta": True,
            "currency_format": "2 490 ₸",
            "numbers": "thousands space, decimal comma",
            "cta_verbs": ["Открыть","Оформить","Настроить","Посмотреть"]
        },
        "hints": hints
    }
    return json.dumps(data, ensure_ascii=False)

def _openai_chat(messages: List[Dict[str, str]], model: str, temperature: float = 0.4, timeout: int = 30) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не установлен")

    url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def _ollama_generate(prompt: str, model: str, url: str, temperature: float = 0.4, timeout: int = 60) -> str:
    api = url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "temperature": temperature, "stream": False}
    resp = requests.post(api, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")

def generate_push_ai(
    product: str,
    ctx: Dict[str, any],
    benefit_kzt: float,
    llm: str = "openai",
    model: str = "gpt-4o-mini",
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.4,
    tries: int = 2,
) -> str:
    """
    Возвращает готовый текст пуша.
    - llm: 'openai' или 'ollama'
    - model: имя модели (напр. gpt-4o-mini / llama3:8b-instruct / qwen2.5:7b-instruct)
    """
    system = build_system_prompt()
    user = build_user_prompt(product, ctx, benefit_kzt)

    last_err = None
    for _ in range(max(1, tries)):
        try:
            if llm == "openai":
                content = _openai_chat(
                    messages=[{"role":"system","content":system},{"role":"user","content":user}],
                    model=model, temperature=temperature
                )
            elif llm == "ollama":
                prompt = (
                    system + "\n\nПОЛЬЗОВАТЕЛЬСКИЕ ДАННЫЕ:\n" + user +
                    "\n\nОтветьте ТОЛЬКО JSON: {\"text\":\"...\"}"
                )
                content = _ollama_generate(prompt, model=model, url=ollama_url, temperature=temperature)
            else:
                raise RuntimeError(f"Неизвестный провайдер LLM: {llm}")

            data = json.loads(content)
            text = str(data.get("text", "")).strip()
            if not text:
                raise ValueError("Пустой text")
            return clamp_policies(text, 220)

        except Exception as e:
            last_err = e
            time.sleep(0.3)

    raise RuntimeError(f"LLM generation failed: {last_err}")
