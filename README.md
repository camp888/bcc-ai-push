# BCC AI 2025 — Personalized Push Notifications

## Установка и запуск

### 1. Подготовка Python и venv
macOS (рекомендуется Python 3.11):
```bash
# Установить python через Homebrew (если не стоит)
brew install python@3.11

# Создать и активировать venv в корне проекта
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Установка зависимостей
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install pandas numpy
```

### 3. Запуск
```bash
python push_generator.py --data-dir "case 1" --out output.csv
```