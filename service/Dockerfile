FROM python:3.10-slim

WORKDIR /app

# Системные зависимости (curl для healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Зависимости и файл окружения
COPY requirements.txt .
COPY dot_env .env

RUN pip install --no-cache-dir -r requirements.txt

# Код сервиса
COPY kbs/ ./kbs/
RUN touch ./kbs/__init__.py

# Безопасность: запуск от non-root пользователя
RUN useradd -m kbsuser && chown -R kbsuser:kbsuser /app
USER kbsuser

EXPOSE 8000

# Запуск с таймаутом keep-alive 600s для долгих /embed запросов
#CMD ["uvicorn", "kbs.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]
CMD ["uvicorn", "kbs.main:app", "--host", "127.0.0.1", "--port", "8000", "--timeout-keep-alive", "600"]
