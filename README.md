
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
- **Индексы:** `tenant_id` (keyword, is_tenant=True), `base_id` (keyword)
- **Payload:** 
  | Поле | Тип | Описание |
  |------|-----|----------|
  | `tenant_id` | string | Идентификатор владельца (JWT claim) |
  | `base_id` | string | Уникальный ID базы знаний |
  | `text` | string | Текст чанка (с инъекцией заголовков `H1 > H2 > H3`) |
  | `doc_id` | string | ID исходного документа |
  | `chunk_index` | integer | Порядковый номер чанка |
  | `created_at` | float | Unix-время загрузки (в секундах с дробной частью) |

### 2. Мета-коллекция (`kb_metadata`)
Хранит конфигурации БЗ, маппинг документов и счётчики квот.
- **Имя:** `kb_metadata`
- **Вектор:** 1D (`[0.0]`), метрика `DOT` (поиск только по payload)
- **HNSW:** `m=0`, `payload_m=16`, `on_disk=True`
- **Индексы:** `tenant_id` (keyword, is_tenant=True), `base_id` (keyword)
- **Payload:**
  | Поле | Тип | Описание |
  |------|-----|----------|
  | `type` | string | Тип записи. В текущей реализации - всегда "kb_config". |
  | `tenant_id` | string | Идентификатор владельца (из JWT). 🔑 Обязательный фильтр для изоляции. |
  | `base_id` | string | Уникальный ID базы знаний. |
  | `collection_name` | string | Имя связанной векторной коллекции (например, `emb_text_embedding_3_small`). |
  | `documents` | object | Реестр загруженных документов (маппинг `doc_id` → наименование). |
  | `embedding_config` | object | Параметры используемого эмбеддера (имя, размерность, url). |
  | `size_kb` | number | Текущий объём БЗ в КБ. Используется для проверки квот перед загрузкой. |
  | `created_at` | float | Unix-время создания конфигурации. |
  | `updated_at` | float | Unix-время последнего обновления или пополнения БЗ. |


### 3. Реестр моделей (`kb_models`)
Централизованное хранение калибровок и эталонных распределений.
- **Имя:** `kb_models`
- **Вектор:** 1D (`[0.0]`), метрика `DOT` (поиск только по payload)
- **HNSW:** `m=0`, `payload_m=16`, `on_disk=True`
- **Индексы:** `model_name` (keyword)
- **Payload:**
  | Поле | Тип | Описание |
  |------|-----|----------|
  | `model_name` | string | Нормализованное имя эмбеддера. |
  | `c_min` | float | Нижняя граница калибровки. *На уровне кода зафиксирована `0.0` (физический нижний предел relevance).* |
  | `c_max` | float | Верхняя граница калибровки. Rolling-максимум relevance. |
  | `calib_version` | integer | Версия структуры калибровки. Инкрементируется при перекалибровке. |
  | `baseline_hist` | array | Гистограмма распределения значений relevance. Используется как база для расчёта PSI (дрейфа). |
  | `total_scores` | integer | Счётчик обработанных скорингов, на которых построена `baseline_hist`. |
  | `updated_at` | float | Unix-время последнего пересчёта параметров калибровки. |


---

## Структура проекта
```
.
├── docker-compose.yml
├── dot_env                  # Образец для .env
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
│       │   ├── grafana-kbs-dashboard.json
│       │   └── qdrant-dashboard.json
│       └── datasources/
│           └── datasources.yml
├── promtail/
│   └── promtail-config.yml
├── qdrant/
│   └── config/
│       └── config.yaml
└── logs/                    # Для отладки
```

---

## Deployment Notes

### 1. Подготовка
```bash
cp dot_env .env
# Заполните: JWT_SECRET, QDRANT_API_KEY  
# Для эмбеддера по умолчанию укажите: EMBEDDER_URL, EMBEDDER_MODEL, EMBEDDER_API_KEY  
# Для доступа к панелям мониторинга:  GRAFANA_USER, GRAFANA_PASSWORD
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


### Аутентификация

Тип токена в заголовке `Authorization` — **JWT (Bearer token)**. JWT должен содержать в claims:
- `tenant_id` — идентификатор клиента.
- `kb_quotas` — информация о лимитах на объем хранения.
- `exp` — срок действия токена.


### Создание/наполнение базы знаний
```
POST /api/v1/kb/embed
```
**Заголовки:**
- `Authorization: Bearer <JWT>`
- `X-Embedding-API-Key: <key>` (API-ключ эмбеддера)

**Тело запроса:**
```json
{
    "openai_api_key": "(опционально) вместо X-Embedding-API-Key",
    "base_id": "uuid (опционально, если не указан — создаётся новая БЗ)",
    "prefix": "kb_",
    "chunk_size": 500,
    "chunk_overlap": 80,
    "separator": "--- BLOCK START ---",
    "file_types": ["список", "допустимых расширений", "имен файлов", "для загрузки"],
    "files_urls": ["(опционально)", "список", "url", "файлов", "для загрузки"],
    "text": "(опционально) дополнительный текст для загрузки",
    "url": "(опционально) - .html для рекурсивного извлечения текста, если не задан text",
    "embedder_type": "platform",
    "embedder_url": "(опционально) для embedder_type: custom",
    "embedder_model": "text-embedding-3-small",
    "embedder_dim": 1536
}
```

**Логика обработки:**
1. Валидация JWT, извлечение tenant_id.
2. Проверка принадлежности БЗ клиенту (если `base_id` указан).
3. Если БЗ новая — создание записи в мета-коллекции `kb_metadata` и основной коллекции в Qdrant.
4. Чанкование документов.
5. Векторизация чанков указанным эмбеддером.
6. Сохранение векторов и payload в основную коллекцию.
7. Обновление словаря `documents` в мета-коллекции.

**Ответ:**
```json
{
    "base_id": "emb_db:uuid",
    "chunks": 30,
    "tokens": 15000,
    "cost_usd": 0.0003,
    "status": "success"
}
```

### Семантический поиск
```
POST /api/v1/kb/query
```
**Заголовки:**
- `Authorization: Bearer <JWT>`
- `X-Embedding-API-Key: <key>` (API-ключ эмбеддера)

**Тело запроса:**
```json
{
    "openai_api_key": "(опционально) вместо X-Embedding-API-Key",
    "base_id": "[emb_db:]uuid",
    "base_ids": ["список", "base_id", "для", "Agentic RAG"],
    "question": "текст запроса",
    "k": 3,
    "relevance_threshold": 0.65
}
```

**Примечания:**
- `base_ids` — LLM может запросить параллельный поиск по нескольким БЗ 
- `relevance_threshold` — порог релевантности в нормализованном диапазоне 0-1.

**Логика:**
1. Валидация JWT, извлечение tenant_id.
2. Поиск по `base_id` в коллекции `kb_metadata` и проверка принадлежности БЗ клиенту.
3. Векторизация запроса соответствующим эмбеддером.
4. Поиск в соответствующей коллекции.

**Ответ:**
```json
{
    "results": [
        {
            "base_id": "emb_db:uuid",
            "id": "chunk_id",
            "page_content": "текст чанка",
            "metadata": {
                "doc_id": "идентификатор исходного документа",
                "source": "имя исходного документа",
                "page": "- для совместимости",
            },
            "relevance": 0.92,
            "raw_distance": 0.16,
        }
    ],
    "tokens": 150,
    "cost_usd": 0.0003,
    "status": "success"
}
```

**Примечания к relevance:**
- `relevance` — нормализованная релевантность [0-1] (1 + cos)/2. 
- `raw_distance` — исходное значение косинусного сходства [-1,1] (cos) из векторной БД.

### Удаление базы знаний
```
POST /api/v1/kb/remove
```
**Заголовки:**
- `Authorization: Bearer <JWT>`

**Тело запроса:**
```json
{
    "base_id": "[emb_db:]uuid",
}
```
**Логика:**
1. Валидация JWT, извлечение tenant_id.
2. Проверка принадлежности БЗ клиенту.
3. Удаление коллекции.
4. Удаление записи из мета-коллекции `kb_metadata`.


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

Все переменные в контейнере kbs используют префикс `KBS_`. Приоритет: `OS env` > `.env` > дефолты.

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

---
