
# Knowledge Base Service (KBS)

Production-ready микросервис для управления векторными базами знаний. Обеспечивает безопасную загрузку документов, каскадное чанкование, адаптивную batch-векторизацию, семантический поиск с калибровкой релевантности, квотирование хранилища и мониторинг дрейфа моделей.

> 🐳 Готов к деплою через Docker Compose | 📊 Наблюдаемость из коробки | 🔒 Строгая мультитенантная изоляция

---

## Ключевые принципы
| Принцип | Реализация |
|---------|------------|
| **Stateless** | Не хранит состояние между запросами. Масштабируется горизонтально через `docker compose scale` или K8s HPA. |
| **Strict Tenant Isolation** | `tenant_id` извлекается исключительно из JWT. Каждый запрос к Qdrant содержит обязательный payload-фильтр `tenant_id + base_id`, что исключает утечку данных между клиентами. |
| **Async-Native Pipeline** | Неблокирующий event loop, batch-эмбеддинг, параллельный поиск по нескольким БЗ. |
| **Unified Storage** | Qdrant используется как для векторов, так и для конфигураций (`kb_metadata`) и реестра моделей (`kb_models`). Отсутствует необходимость в отдельной реляционной СУБД. |
| **Auto-Calibration** | Автоматический расчет и калибровка диапазона распределения значений relevance score для каждого эмбеддера на основе попарных косинусных расстояний реальных чанков. |
| **Quota Enforcement** | Синхронная проверка квоты `tenant_quota_kb` (из JWT) перед загрузкой каждого батча. Счётчик `usage_{tenant_id}` обновляется в `kb_metadata`. |
| **Structured Logging** | Каждый лог содержит `request_id`, `tenant_id`, `endpoint`. Парсится Promtail → Loki, фильтруется в Grafana LogQL. |

KBS является stateless-микросервисом, вынесенным за пределы основной платформы для изоляции векторной логики, упрощения масштабирования и гарантии безопасности данных клиентов.

```text
┌─────────────────┐      JWT + Bearer       ┌─────────────────┐
│   Platform /    │ ──────────────────────► │                 │
│   AI-Agents     │                         │      KBS        │
│   (FastAPI)     │ ◄────────────────────── │  (FastAPI +     │
└─────────────────┘      JSON Response      │   Async I/O)    │
                                           └───────┬─────────┘
                                                   │ gRPC/HTTP
                                           ┌───────▼─────────┐
                                           │     Qdrant      │
                                           │ (Vector + Meta) │
                                           └───────┬─────────┘
                                                   │
                    ┌──────────────────────────────┼────────────────────────────┐
                    │                              │ scrape (metrics)           │
                    │                              ▼                            │
                    │                    ┌─────────────────┐                   │
                    │                    │    Prometheus   │                   │
                    │                    │  (Time-Series)  │                   │
                    │                    └────────┬────────┘                   │
                    │                             │ query                      │
                    │                             ▼                            │
                    │                    ┌─────────────────┐                   │
                    │                    │     Grafana     │                   │
                    │                    │ (Dashboards +   │                   │
                    │                    │   Alerting)     │                   │
                    │                    └────────┬────────┘                   │
                    │                             │ query (LogQL)              │
                    │                             ▼                            │
                    │                    ┌─────────────────┐                   │
                    │                    │      Loki       │                   │
                    │                    │  (Log Aggregator)│                  │
                    │                    └────────▲────────┘                   │
                    │                             │ push                       │
                    │                    ┌────────┴────────┐                   │
                    │                    │    Promtail     │                   │
                    │                    │  (Log Shipper)  │                   │
                    │                    └────────▲────────┘                   │
                    │                             │ read                       │
                    └─────────────────────────────┼────────────────────────────┘
                                                  │
                                     ┌────────────┴────────────┐
                                     │   Docker Logs (stdout)  │
                                     │   KBS → JSON structured │
                                     └─────────────────────────┘
```

### Описание потоков данных
| Стрелка | Протокол | Что передаётся |
|---------|----------|----------------|
| `Platform → KBS` | HTTPS + Bearer | Запросы `/embed`, `/query` с JWT |
| `KBS → Qdrant` | gRPC (с fallback на HTTP) | Векторный поиск, upsert, scroll |
| `KBS → Prometheus` | HTTP `/metrics` | Scraping метрик (латентность, RPS, drift) |
| `Prometheus → Grafana` | HTTP API | Визуализация временных рядов, алертинг |
| `Promtail → Loki` | HTTP push | Структурированные логи (`request_id`, `tenant_id`) |
| `Grafana → Loki` | LogQL query | Поиск и фильтрация логов из UI дашборда |

---

## Структура коллекций Qdrant

### 1. Векторные коллекции (`emb_{clean_model}`)
Создаются динамически при загрузке первой БЗ с соответствующим эмбеддером. Коллекция используется всеми тенантами, применяющими соответствующий эмбеддер.
- **Имя:** `emb_{cleaned_model_name}` (например, `emb_text_embedding_3_small`)
- **Метрика:** `COSINE`
- **Payload:** 
  | Поле | Тип | Описание |
  |------|-----|----------|
  | `tenant_id` | string | Идентификатор владельца (JWT claim) |
  | `base_id` | string | Уникальный ID базы знаний |
  | `text` | string | Текст чанка (с инъекцией заголовков `H1 > H2 > H3`) |
  | `doc_id` | string | ID исходного документа |
  | `chunk_index` | integer | Порядковый номер чанка |
  | `created_at` | float | Unix-время загрузки |

### 2. Мета-коллекция (`kb_metadata`)
Хранит конфигурации БЗ, маппинг документов и счётчики квот.
- **Имя:** `kb_metadata`
- **Вектор:** 1D (`[0.0]`), метрика `DOT` (поиск только по payload)
- **HNSW:** `m=1`, `payload_m=16`, `on_disk=True`
- **Индексы:** `tenant_id` (keyword), `base_id` (keyword)
- **Payload (типы записей):**
  | Тип точки | Ключевые поля | Назначение |
  |-----------|---------------|------------|
  | `kb_config` (`id = base_id`) | `tenant_id` 🔑, `base_id`, `collection_name`, `documents`, `embedding_config`, `size_kb`, `created_at`, `updated_at` | Конфигурация БЗ. Поле `size_kb` используется для динамического расчёта квоты. |

### 3. Реестр моделей (`kb_models`)
Централизованное хранение калибровок и эталонных распределений.
- **Имя:** `kb_models`
- **ID точки:** `{model_name}` (например, `text_embedding_3_small`)
- **Payload:** `c_min`, `c_max`, `is_calibrated`, `baseline_hist`, `method`, `fitted_at`
- **Назначение:** Автокалибровка, дрифт-мониторинг (PSI), быстрый lookup параметров нормализации.

---

## Структура проекта
```
.
├── docker-compose.yml
├── .env.example
├── service/
│   ├── Dockerfile
│   ├── dot_env              # Копируется в контейнер как /app/.env
│   ├── requirements.txt
│   └── kbs/
│       └── main.py          # Ядро: API, middleware, бизнес-логика
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── provisioning/
│       ├── dashboards/
│       │   └── kbs-dashboard.json
│       └── datasources/
│           └── datasources.yml
├── promtail/
│   └── promtail-config.yml
├── qdrant/
│   └── config/
│       └── config.yaml
└── logs/                    # Маппится в контейнер для отладки
```

---

## Deployment Notes

### 1. Подготовка
```bash
cp .env.example .env
# Заполните: KBS_JWT_SECRET, KBS_DEFAULT_API_KEY, KBS_QDRANT_API_KEY
```

### 2. Запуск
```bash
docker compose up -d --build
```
Все сервисы поднимутся в изолированной сети `kbs_net`. Коллекции и индексы инициализируются автоматически при старте KBS.

### 3. ⚠️ Настройка таймаутов для `/embed`
Векторизация больших файлов может занимать до 10 минут.
- Uvicorn: `--timeout-keep-alive 600` (уже в `Dockerfile`)
- Reverse Proxy (Nginx/Ingress): добавьте явно:
  ```nginx
  proxy_read_timeout 600s;
  proxy_send_timeout 600s;
  ```

### 4. Healthcheck & Масштабирование
```bash
curl -f http://localhost:8000/api/v1/kb/health
# → {"status":"ok","uptime":"running","version":"1.0.0"}

docker compose up -d --scale kbs=3  # Stateless, работает без конфликтов
```

---

## API & Swagger

| Интерфейс | URL |
|-----------|-----|
| Swagger UI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |
| OpenAPI JSON | `http://localhost:8000/openapi.json` |

🔒 **Аутентификация:** `Authorization: Bearer <JWT>`  
🔑 **Кастомные эмбеддеры:** `X-Embedding-API-Key: <key>`

| Метод | Путь | Назначение |
|-------|------|------------|
| `POST` | `/api/v1/kb/embed` | Загрузка/пополнение БЗ, чанкование, векторизация |
| `POST` | `/api/v1/kb/query` | Семантический поиск (поддержка `base_ids` для Agentic RAG) |
| `POST` | `/api/v1/kb/remove` | Удаление БЗ и очистка метаданных |
| `GET`  | `/api/v1/kb/health` | Проверка работоспособности |
| `GET`  | `/metrics` | Prometheus-метрики |

---

## Мониторинг и Наблюдаемость

### Prometheus-метрики
| Метрика | Тип | Описание |
|---------|-----|----------|
| `kbs_requests_total{endpoint, status}` | Counter | RPS и распределение ответов |
| `kbs_embed_duration_seconds` | Histogram | Латентность пайплайна `/embed` |
| `kbs_query_duration_seconds{model_key}` | Histogram | Латентность `/query` per модель |
| `kbs_score_distribution_bucket{model_key}` | Histogram | Распределение калиброванной релевантности |
| `kbs_calibration_out_of_range_total{model_key}` | Counter | Число скоров за пределами `[c_min, c_max]` |
| `kbs_drift_psi{model_key}` | Gauge | Population Stability Index (дрейф распределения) |
| `kbs_score_mean{model_key}` | Gauge | Скользящее среднее `raw_cosine` |
| `kbs_score_out_of_range_ratio{model_key}` | Gauge | Доля скоров вне калибровочного диапазона |

### Grafana & Loki
- **Дашборд:** Авто-провижинится при старте. Содержит разделы: Overview, Score Distribution, Drift Monitoring, Structured Logs.
- **Data Links:** Клик на любую точку графика открывает **Loki Explore** с предзаполненным LogQL и тем же тайм-рейнджем.
- **Unified Alerting:** Встроен в Grafana. Не требует Alertmanager. Правила создаются через UI или YAML-провижининг.

### LogQL-фильтры (примеры)
```text
{container="kbs"} | json | level="ERROR" | tenant_id="abc-123"
{container="kbs"} | json | request_id="f47ac10b-58cc-..." | line_format `{{.msg}}`
```

---

## Environment Variables

Все переменные используют префикс `KBS_`. Приоритет: `OS env` > `.env` > дефолты.

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `KBS_QDRANT_URL` | `http://qdrant:6333` | Адрес Qdrant (gRPC автоматически) |
| `KBS_QDRANT_API_KEY` | `None` | Ключ аутентификации Qdrant |
| `KBS_DEFAULT_API_KEY` | `None` | API-ключ платформенного эмбеддера |
| `KBS_JWT_SECRET` | **Обязательно** | Секрет для валидации JWT |
| `KBS_DEFAULT_QUOTA_KB` | `16384` (16 MB) | Квота на тенанта (fallback, если нет в JWT) |
| `KBS_COLLECTION_PREFIX` | `emb_` | Префикс имён векторных коллекций |
| `KBS_LEGACY_QUERY_URL` | `None` | URL legacy `/query`. Если задан → включается миграционный прокси |
| `KBS_LOG_LEVEL` | `INFO` | Уровень логирования (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `KBS_UVICORN_TIMEOUT_KEEP_ALIVE` | `600` | Таймаут keep-alive для длительных `/embed` |
| `KBS_EMBED_BATCH_SIZE_KB` | `64` | Целевой размер батча эмбеддинга в КБ (адаптивный батчинг) |
| `GRAFANA_USER` / `GRAFANA_PASSWORD` | `admin` / `admin` | Учётные данные Grafana UI |

---
