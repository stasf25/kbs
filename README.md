# Knowledge Base Service (KBS)

Микросервис управления базами знаний, обеспечивающий безопасное хранение, векторизацию и семантический поиск документов. Построен на **FastAPI + Qdrant** с строгой мультитенантной изоляцией, поддержкой кастомных эмбеддеров и встроенным стеком мониторинга.

---

## 🏗 Архитектура

KBS является **stateless**-микросервисом, вынесенным за пределы основной платформы для изоляции векторной логики, упрощения масштабирования и гарантии безопасности данных клиентов.

```
┌─────────────────┐      JWT + Bearer       ┌─────────────────┐
│   Platform /    │ ──────────────────────► │                 │
│   AI-Agents     │                         │      KBS        │
│   (FastAPI)     │ ◄────────────────────── │  (FastAPI +     │
└─────────────────┘      JSON Response      │   Async I/O)    │
                                           └────────┬────────┘
                                                    │ gRPC/HTTP
                                           ┌────────▼────────┐
                                           │      Qdrant     │
                                           │  (Vector Store  │
                                           │   + Meta-Store) │
                                           └─────────────────┘
```

### Ключевые принципы
- **Stateless**: Не хранит состояние между запросами. Масштабируется горизонтально через K8s HPA или Docker replicas.
- **Strict Tenant Isolation**: `tenant_id` извлекается исключительно из JWT. Каждый запрос к Qdrant содержит обязательный payload-фильтр `tenant_id + base_id`. Физическое разделение коллекций исключает cross-tenant leakage.
- **Async-Native Pipeline**: Неблокирующий event loop, batch-эмбеддинг, параллельный поиск по нескольким БЗ (`asyncio.gather`).
- **Unified Storage**: Qdrant используется как для векторов, так и для конфигураций БЗ (`kb_metadata`). Отсутствует необходимость в отдельной реляционной СУБД.
- **Calibrated Relevance**: Сырые косинусные скоры нормализуются по формуле `(1 + cos) / 2` и маппятся на общую шкалу `[0, 1]` через калибровочные параметры, хранящиеся в `kb_metadata`.

---

## 🗄 Структура коллекций Qdrant

### 1. Основная коллекция БЗ
Создаётся для каждой базы знаний клиента.
- **Имя:** `kb_{tenant_id}_{base_id}`
- **Метрика:** `COSINE`
- **Payload:**
  | Поле | Тип | Описание |
  |------|-----|----------|
  | `tenant_id` | string | Идентификатор владельца (JWT claim) |
  | `base_id` | string | Уникальный ID базы знаний |
  | `text` | string | Текст чанка (с инъекцией заголовков `H1 > H2 > H3`) |
  | `doc_id` | string | ID исходного документа |
  | `chunk_index` | integer | Порядковый номер чанка |
  | `created_at` | datetime | Время загрузки |

### 2. Мета-коллекция (`kb_metadata`)
Хранит конфигурации всех БЗ, параметры эмбеддеров и маппинг документов.
- **Имя:** `kb_metadata`
- **Вектор:** 1D (`[0.0]`), метрика `DOT` (поиск только по payload)
- **HNSW:** `m=1`, `on_disk=True`, `memmap_threshold=1`, `full_scan_threshold=1`
- **Индексы:** `tenant_id` (keyword), `base_id` (keyword)
- **Payload:**
  | Поле | Тип | Описание |
  |------|-----|----------|
  | `type` | string | `"kb_config"` |
  | `tenant_id` | string | Владелец БЗ |
  | `base_id` | string | ID базы |
  | `collection_name` | string | Связь с основной коллекцией |
  | `documents` | object | `{doc_id: doc_name}` |
  | `embedding_config` | object | `type`, `model_name`, `dimension`, `url`, `calibration_params` |
  | `disk_usage` | integer | Занятое место (байты) |
  | `created_at` / `updated_at` | datetime | Временные метки |

---

## 📦 Структура проекта

```
.
├── service/
│   ├── Dockerfile             # Сборка образа KBS
│   ├── requirements.txt       # Python-зависимости
│   └── kbs/
│       └── main.py            # Ядро микросервиса (FastAPI + логика)
├── prometheus/
│   └── prometheus.yml         # Конфиг скрапинга метрик KBS & Qdrant
├── grafana/
│   └── provisioning/
│       └── dashboards/
│           ├── dashboards.yml # Авто-подключение дашбордов
│           └── grafana-kbs-dashboard.json
├── qdrant/
│   └── config/
│       └── config.yaml        # Минимальный конфиг Qdrant
├── logs/                      # Структурированные JSON-логи приложения
├── docker-compose.yml         # Оркестрация стека
├── .env                       # Секреты и настройки (не коммитится!)
└── README.md
```

---

## 🚀 Deployment Notes

### 1. Подготовка
```bash
# 1. Скопируйте окружение
cp .env.example .env

# 2. Заполните обязательные переменные
# QDRANT_API_KEY, EMBEDDING_API_KEY, JWT_SECRET, GRAFANA_USER/PASSWORD
```

### 2. Запуск стека
```bash
docker compose up -d --build
```
Сервисы запустятся в изолированной сети `kbs_net`. Qdrant будет проинициализирован автоматически при первом старте KBS.

### 3. ⚠️ Важное: Настройка таймаутов для `/embed`
Операция загрузки и векторизации документов выполняется синхронно и может занимать до 5 минут для больших файлов.
- **Uvicorn** уже настроен на `--timeout-keep-alive 600`.
- **Обратный прокси (Nginx / Ingress / Cloudflare)** требует явной настройки:
  ```nginx
  proxy_read_timeout 600s;
  proxy_send_timeout 600s;
  ```
  Без этого прокси разорвёт соединение до завершения обработки, несмотря на рабочий Keep-Alive.

### 4. Healthcheck
```bash
curl -f http://localhost:8000/api/v1/kb/health
# Ожидаемый ответ: {"status":"ok","uptime":"running - для совместимости","version":"1.0.0"}
```

### 5. Масштабирование
KBS полностью stateless. Для горизонтального масштабирования:
```bash
docker compose up -d --scale kbs=3
```
Рекомендуется вынести Qdrant в кластерный режим (Qdrant Enterprise/Cloud) при нагрузке >10k RPS или >500M точек.

---

## 📖 API & Swagger

Сервис автоматически генерирует OpenAPI-спецификацию.

| Интерфейс | URL |
|-----------|-----|
| **Swagger UI** | `http://localhost:8000/docs` |
| **ReDoc** | `http://localhost:8000/redoc` |
| **OpenAPI JSON** | `http://localhost:8000/openapi.json` |

### Аутентификация в Swagger
1. Нажмите 🔒 **Authorize** в правом верхнем углу.
2. Введите: `Bearer <your_jwt_token>`
3. Для кастомных эмбеддеров передавайте ключ в заголовке `X-Embedding-API-Key` прямо в интерфейсе (раздел `Try it out` → `Headers`).

### Основные эндпоинты (v1)
| Метод | Путь | Назначение |
|-------|------|------------|
| `POST` | `/api/v1/kb/embed` | Создание/пополнение БЗ, чанкование, векторизация |
| `POST` | `/api/v1/kb/query` | Семантический поиск (поддержка multi-`base_id` для Agentic RAG) |
| `POST` | `/api/v1/kb/remove` | Удаление БЗ и её коллекции |
| `GET`  | `/api/v1/kb/health` | Проверка работоспособности |
| `GET`  | `/metrics` | Prometheus-метрики |

---

## 📊 Мониторинг

Стек включает **Prometheus** (сбор метрик) и **Grafana** (визуализация). Дашборд деплоится автоматически через provisioning.

- **Grafana:** `http://localhost:3000` (логин/пароль из `.env`)
- **Prometheus:** `http://localhost:9090`

### Ключевые метрики
| Метрика | Тип | Описание |
|---------|-----|----------|
| `kbs_requests_total` | Counter | Общее число запросов по эндпоинтам и статусам |
| `kbs_embed_duration_seconds` | Histogram | Латентность пайплайна `/embed` |
| `kbs_query_duration_seconds` | Histogram | Латентность `/query` (с разбивкой по `model_key`) |
| `kbs_score_distribution` | Histogram | Распределение нормализованной релевантности per model |
| `kbs_ece_per_model` | Gauge | Expected Calibration Error (для алертинга на дрейф) |
| `qdrant_collections_count` / `qdrant_points_count_total` | Gauge | Состояние векторного хранилища |

---

## 🔑 Environment Variables

| Переменная | Описание | Пример |
|------------|----------|--------|
| `QDRANT_URL` | Адрес инстанса Qdrant | `http://qdrant:6333` |
| `QDRANT_API_KEY` | Ключ доступа к Qdrant | `qdrant-sec-...` |
| `EMBEDDING_API_KEY` | API-ключ эмбеддера по умолчанию (OpenAI) | `sk-proj-...` |
| `JWT_SECRET` | Секрет для валидации Bearer-токенов | `your-jwt-secret` |
| `LOG_LEVEL` | Уровень логирования | `INFO` или `DEBUG` |
| `GRAFANA_USER` | Администратор Grafana | `admin` |
| `GRAFANA_PASSWORD` | Пароль администратора Grafana | `secure-password` |

---
