# main.py
"""
Knowledge Base Service (KBS) v1.0.0
Микросервис управления базами знаний на FastAPI + Qdrant
Соответствует ТЗ KB_tz_v00.md + Архитектурные уточнения v1.1
"""
import os
import time
import asyncio
import logging
import json
import httpx
import grpc
import jwt
import uuid
import math
import random
import contextvars
import numpy as np
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Depends, Header, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field, SettingsConfigDict
from pydantic_settings import BaseSettings
from dotenv import find_dotenv
from qdrant_client import AsyncQdrantClient, models
from openai import AsyncOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_kreuzberg import KreuzbergLoader
from kreuzberg import ExtractionConfig
from prometheus_client import Histogram, Counter, Gauge, REGISTRY, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️ Конфигурация сервиса (единый источник настроек)
# ─────────────────────────────────────────────────────────────────────────────
class KBSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KBS_",
        env_file=find_dotenv(raise_error_if_not_found=False, usecwd=True),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: Optional[str] = None

    default_embedder_url: str = "https://api.openai.com/v1"
    default_embedder_model: str = "text-embedding-3-small"
    default_embedder_dim: int = 1536
    #default_embedder_url: str = "https://openrouter.ai/api/v1/embeddings"
    #default_embedder_model: str = "baai/bge-m3"
    #default_embedder_dim: int = 1024
    default_api_key: Optional[str] = None

    jwt_secret: str
    jwt_algorithm: str = "HS256"

    default_tokens_price: float = 0.02   # for 1M tokens
    #pricing: Dict[str, float] = {"text-embedding-3-small": 0.02, "bge-m3": 0.01}

    default_chunk_size: int = 500
    default_chunk_overlap: int = 80
    default_separator: str = "--- BLOCK START ---"

    collection_prefix: str = "emb_"
    meta_collection_name: str = "kb_metadata"
    models_collection_name: str = "kb_models"

    calibration_default_min: float = 0.0
    calibration_default_max: float = 1.0
    default_quota_kb: int = 5120  # 5 MB по умолчанию

    log_level: str = "INFO"
    uvicorn_timeout_keep_alive: int = 600

    legacy_query_url: Optional[str] = None          # URL legacy-эндпоинта (/query)
    legacy_proxy_timeout: float = 30.0              # Таймаут прокси-запроса

    @classmethod
    def from_env(cls) -> "KBSettings":
        return cls()

# ─────────────────────────────────────────────────────────────────────────────
# 📝 Логирование с контекстом (ТЗ 7.3: timestamp, level, request_id, tenant_id, endpoint)
# ─────────────────────────────────────────────────────────────────────────────
request_id_var = contextvars.ContextVar("request_id", default="-")
tenant_id_var = contextvars.ContextVar("tenant_id", default="-")
endpoint_var = contextvars.ContextVar("endpoint", default="-")

class ContextLogFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        record.tenant_id = tenant_id_var.get()
        record.endpoint = endpoint_var.get()
        return True

logging.basicConfig(
    level=os.getenv("KBS_LOG_LEVEL", "INFO").upper(),
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","service":"kbs","request_id":"%(request_id)s","tenant_id":"%(tenant_id)s","endpoint":"%(endpoint)s","msg":"%(message)s"}'
)
logger = logging.getLogger("kbs")
logging.getLogger().addFilter(ContextLogFilter())

# ─────────────────────────────────────────────────────────────────────────────
# 📊 Prometheus Metrics
# ─────────────────────────────────────────────────────────────────────────────
METRICS_PREFIX = "kbs_"
EMBED_LATENCY = Histogram(f"{METRICS_PREFIX}embed_duration_seconds", "Embedding pipeline latency")
QUERY_LATENCY = Histogram(f"{METRICS_PREFIX}query_duration_seconds", "Query pipeline latency", ["model_key"])
REQ_COUNTER = Counter(f"{METRICS_PREFIX}requests_total", "Total API requests", ["endpoint", "status"])
SCORE_DIST = Histogram(f"{METRICS_PREFIX}score_distribution", "Calibrated score distribution", ["model_key"], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Дрифт-метрики (PSI, mean, OOR ratio)
DRIFT_PSI_GAUGE = Gauge(f"{METRICS_PREFIX}drift_psi", "Population Stability Index vs baseline", ["model_key"])
SCORE_MEAN_GAUGE = Gauge(f"{METRICS_PREFIX}score_mean", "Rolling mean of raw cosine", ["model_key"])
SCORE_OOR_RATIO_GAUGE = Gauge(f"{METRICS_PREFIX}score_out_of_range_ratio", "Ratio of scores outside calibrated range", ["model_key"])

# ─────────────────────────────────────────────────────────────────────────────
# 📦 Pydantic Models (ТЗ 5.3)
# ─────────────────────────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    openai_api_key: Optional[str] = None
    base_id: Optional[str] = None
    prefix: str = "kb_"
    chunk_size: int = 500
    chunk_overlap: int = 80
    separator: str = "--- BLOCK START ---"
    file_types: List[str] = ["pdf", "txt", "md", "html"]
    files_urls: Optional[List[str]] = None
    yandex_disk_folder: Optional[str] = None
    text: Optional[str] = None
    url: Optional[str] = None
    record_id: Optional[str] = "temp_record"
    email: Optional[str] = "unknown"
    embedder_type: str = "platform"
    embedder_url: Optional[str] = None
    embedder_model: str = "text-embedding-3-small"
    embedder_dim: int = 1536

class EmbedResponse(BaseModel):
    base_id: str
    chunks: int
    tokens: int
    cost_usd: float
    status: str = "success"

class QueryRequest(BaseModel):
    openai_api_key: Optional[str] = None
    embedder_url: Optional[str] = None  # Позволяет сменить провайдера на лету
    base_id: Optional[str] = None
    base_ids: Optional[List[str]] = None
    question: str = Field(..., min_length=1)
    k: int = Field(3, ge=1, le=50)
    relevance_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

class ChunkMetadata(BaseModel):
    doc_id: str
    source: str
    page: str = "-"

class ChunkResult(BaseModel):
    base_id: str
    id: str
    page_content: str
    metadata: ChunkMetadata
    relevance: float
    raw_distance: float

class QueryResponse(BaseModel):
    results: List[ChunkResult]
    tokens: int = 0
    cost_usd: float = 0.0
    status: str = "success"

class RemoveRequest(BaseModel):
    base_id: str

class HealthResponse(BaseModel):
    status: str
    uptime: str = "running"
    version: str = "1.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# 🧩 Каскадный чанкер
# ─────────────────────────────────────────────────────────────────────────────
def cascade_chunk_markdown(
    markdown_text: str,
    doc_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    separator: str = "--- BLOCK START ---"
) -> List[Dict[str, Any]]:
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")],
        strip_headers=True
    )
    header_docs = md_splitter.split_text(markdown_text)

    enriched = []
    for doc in header_docs:
        headers = [doc.metadata.get(h) for h in ["H1", "H2", "H3", "H4"] if doc.metadata.get(h)]
        prefix = " > ".join(headers) + "\n" if headers else ""
        content = f"{prefix}{doc.page_content.strip()}"
        if content.strip():
            enriched.append(Document(page_content=content, metadata={"doc_id": doc_id}))

    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[separator, "\n\n", "\n", ". ", " ", ""],
        keep_separator="end"
    )
    final_chunks = rec_splitter.split_documents(enriched)

    return [
        {"id": f"{doc_id}#chunk_{i}", "text": chunk.page_content, "doc_id": doc_id,
         "chunk_index": i, "created_at": time.time()}
        for i, chunk in enumerate(final_chunks) if chunk.page_content.strip()
    ]

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Middleware для сквозного логирования (ТЗ 7.3)
# ─────────────────────────────────────────────────────────────────────────────
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_var.set(req_id)
        endpoint_var.set(f"{request.method} {request.url.path}")
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Middleware для проксирования запросов на legacy endpoint
# ─────────────────────────────────────────────────────────────────────────────
class LegacyProxyMiddleware(BaseHTTPMiddleware):
    """
    Проксирует запросы к старым БЗ на legacy-эндпоинт.
    Если base_id найден в kb_metadata → обработка в новом сервисе.
    Если base_id отсутствует → запрос проксируется на legacy_url.
    """
    def __init__(self, app, legacy_url: str, timeout: float = 30.0):
        super().__init__(app)
        self.legacy_url = legacy_url
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        # Работаем только с POST /api/v1/kb/query и если legacy_url задан
        if request.method != "POST" or request.url.path != "/api/v1/kb/query":
            return await call_next(request)
        
        settings = request.app.state.settings
        if not settings.legacy_query_url:
            return await call_next(request)

        # Читаем body один раз
        body_bytes = await request.body()
        
        try:
            payload = json.loads(body_bytes)
            target_id = payload.get("base_id") or (payload.get("base_ids") or [None])[0]
            
            if target_id:
                # O(1) проверка существования base_id в мета-коллекции
                backend = request.app.state.kbs.backend
                found = await backend.retrieve(
                    collection_name=settings.meta_collection_name, 
                    ids=[target_id]
                )
                if not found:
                    # Базы нет в Qdrant → проксируем на legacy
                    return await self._proxy_to_legacy(request, body_bytes)
        except Exception:
            pass  # Fail-open: при любой ошибке идём в новый сервис

        # Восстанавливаем body для call_next()
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive
        
        return await call_next(request)

    async def _proxy_to_legacy(self, request: Request, body: bytes):
        """Прямой прокси-запрос с сохранением заголовков"""
        headers = {k: v for k, v in request.headers.items() 
                   if k.lower() not in ("host", "content-length", "transfer-encoding")}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.legacy_url, content=body, headers=headers)
        return StarletteResponse(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Stateless Service Class (DI + Lifecycle)
# ─────────────────────────────────────────────────────────────────────────────
class KBSService:
    def __init__(self, backend: AsyncQdrantClient, settings: KBSettings):
        self.backend = backend
        self.settings = settings
        self._meta_collection = settings.meta_collection_name
        self._models_collection = settings.models_collection_name
        
        # Дрифт-трекер (in-memory)
        self._score_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._baselines: Dict[str, np.ndarray] = {}
        self._bucket_edges = np.linspace(0.0, 1.0, 11)
        self._lock = asyncio.Lock()
        
        self.out_of_range_counter = Counter(
            f"{METRICS_PREFIX}calibration_out_of_range_total",
            "Count of scores outside calibrated range",
            ["model_key"]
        )

    def _get_clean_model_key(self, model: str) -> str:
        base = model.split(":", 1)[0].strip()
        return base.replace("/", "_").replace(" ", "_").replace(".", "_")

    def _get_collection_name(self, model: str) -> str:
        return f"{self.settings.collection_prefix}{self._get_clean_model_key(model)}"

    async def _extract_tenant_context(self, auth_header: Optional[str]) -> dict:
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(401, detail="Missing or invalid Bearer token")
        try:
            payload = jwt.decode(
                auth_header.split(" ", 1)[1], self.settings.jwt_secret, algorithms=[self.settings.jwt_algorithm],
                options={"verify_exp": True, "verify_iss": True, "verify_aud": True}
            )
            tenant_id = payload.get("tenant_id") or payload.get("sub")
            if not tenant_id: raise ValueError("tenant_id claim missing")
            tenant_id_var.set(tenant_id)  # Для логирования
            quota_kb = int(payload.get("tenant_quota_kb", self.settings.default_quota_kb))
            return {"tenant_id": tenant_id, "quota_kb": quota_kb}
        except (jwt.PyJWTError, ValueError, TypeError) as e:
            raise HTTPException(401, detail=f"Invalid token: {e}")

    async def _fetch_kb_meta(self, tenant_id: str, base_ids: List[str]) -> Dict[str, Dict]:
        points, _ = await self.backend.scroll(
            collection_name=self._meta_collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                models.FieldCondition(key="base_id", match=models.MatchAny(any=base_ids))
            ]),
            limit=len(base_ids), with_payload=True, with_vectors=False
        )
        found = {p.payload["base_id"]: p.payload for p in points}
        missing = set(base_ids) - found.keys()
        if missing:
            raise HTTPException(403, detail=f"KB not found or access denied: {', '.join(missing)}")
        return found


    async def _delete_base_points(self, collection_name: str, tenant_id: str, base_id: str):
        """
        Удаляет точки БЗ из коллекции.
        - Если коллекции нет: warning + продолжает выполнение (очистка orphan-метаданных).
        - Если ошибка удаления: error + raise → HTTP 500 (требует вмешательства).
        """
        if not collection_name:
            logger.error(
                f"❌ Internal error: collection_name is missing/empty for base_id={base_id}. "
                f"Metadata is corrupted. Aborting deletion."
            )
            raise ValueError("Corrupted metadata: missing collection_name")

        selector = models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                models.FieldCondition(key="base_id", match=models.MatchValue(value=base_id))
            ])
        )
        try:
            await self.backend.delete(collection_name=collection_name, points_selector=selector)
            logger.info(f"✅ Deleted points for base_id={base_id} (tenant={tenant_id}) from {collection_name}")
        except Exception as e:
            # Определяем: отсутствие коллекции или реальная ошибка?
            if ((getattr(e, "status_code", None) == 404) or \
                (callable(getattr(e, "code", None)) and e.code() == grpc.StatusCode.NOT_FOUND)
            ):
                logger.warning(
                    f"⚠️ Collection '{collection_name}' missing for base_id={base_id}. "
                    f"Proceeding with metadata cleanup (orphan reconciliation)."
                )
                return  # Не пробрасываем: даём вызывающему коду удалить запись из kb_metadata
            else:
                logger.error(f"❌ Point deletion failed for base_id={base_id} in {collection_name}:"
                             f" {e}"
                )
                raise  # Пробрасываем → FastAPI вернёт HTTP 500


    async def _get_model_calibration(self, model_key: str) -> Optional[Dict[str, float]]:
        try:
            points = await self.backend.retrieve(collection_name=self._models_collection, ids=[model_key], with_payload=["c_min", "c_max", "is_calibrated"])
            if points and points[0].payload.get("is_calibrated"):
                return {"min": points[0].payload["c_min"], "max": points[0].payload["c_max"]}
        except Exception: pass
        return None

    def _apply_calibration(self, raw_cosine: float, cal_params: Optional[Dict], model_key: str = "unknown") -> float:
        local = max(0.0, min(1.0, (raw_cosine + 1.0) / 2.0))
        if not cal_params:
            return local

        c_min = cal_params.get("min", self.settings.calibration_default_min)
        c_max = cal_params.get("max", self.settings.calibration_default_max)
        
        if local < c_min or local > c_max:
            self.out_of_range_counter.labels(model_key=model_key).inc()

        if c_max - c_min < 1e-6:
            logger.warning(f"Degenerate calibration for {model_key}: fallback to base norm.")
            return local
        return max(0.0, min(1.0, (local - c_min) / (c_max - c_min)))

    def _record_metrics(self, model_key: str, scores: List[float]):
        for s in scores:
            SCORE_DIST.labels(model_key=model_key).observe(s)

    async def _update_score_stats(self, model_key: str, scores: List[float]):
        async with self._lock:
            window = self._score_windows[model_key]
            window.extend(scores)

            if model_key not in self._baselines:
                try:
                    pts = await self.backend.retrieve(collection_name=self._models_collection, ids=[model_key], with_payload=["baseline_hist"])
                    if pts and pts[0].payload.get("baseline_hist"):
                        self._baselines[model_key] = np.array(pts[0].payload["baseline_hist"], dtype=np.float32)
                except Exception: pass

            if self._baselines.get(model_key) is None and len(window) >= 1000:
                hist, _ = np.histogram(list(window), bins=self._bucket_edges, density=False)
                await self.backend.upsert(
                    collection_name=self._models_collection, 
                    points=[models.PointStruct(
                        id=model_key, vector=[0.0], 
                        payload={"baseline_hist": [float(x) for x in hist]}
                    )]
                )
                self._baselines[model_key] = hist.astype(np.float32)

            if self._baselines.get(model_key) is not None and len(window) >= 500:
                current_hist, _ = np.histogram(list(window), bins=self._bucket_edges, density=False)
                baseline = self._baselines[model_key]
                psi = 0.0
                s_curr, s_base = sum(current_hist), sum(baseline)
                for act, exp in zip(current_hist, baseline):
                    act_pct = max(act / s_curr, 1e-6)
                    exp_pct = max(exp / s_base, 1e-6)
                    psi += (act_pct - exp_pct) * np.log(act_pct / exp_pct)
                DRIFT_PSI_GAUGE.labels(model_key=model_key).set(round(psi, 4))
                SCORE_MEAN_GAUGE.labels(model_key=model_key).set(round(np.mean(window), 4))
                oor = sum(1 for s in window if s < self.settings.calibration_default_min or s > self.settings.calibration_default_max)
                SCORE_OOR_RATIO_GAUGE.labels(model_key=model_key).set(round(oor / len(window), 4) if window else 0.0)

    async def _batch_embed(self, model: str, url: str, api_key: str, texts: List[str], is_default: bool):
        if is_default:
            client = AsyncOpenAI(api_key=self.settings.default_api_key, base_url=url or self.settings.default_embedder_url)
            resp = await client.embeddings.create(model=model, input=texts, encoding_format="float")
            tokens = resp.usage.total_tokens
            cost = round((tokens / 1_000_000) * self.settings.default_tokens_price, 6)
            return [d.embedding for d in resp.data], tokens, cost
        else:
            embedder = AsyncOpenAI(api_key=api_key, base_url=url)
            resp = await embedder.embeddings.create(model=model, input=texts, encoding_format="float")
            return [d.embedding for d in resp.data], 0, 0.0

    async def _detect_embedding_dimension(self, model: str, url: str, api_key: str) -> int:
        client = AsyncOpenAI(api_key=api_key or self.settings.default_api_key, base_url=url)
        try:
            resp = await client.embeddings.create(model=model, input="test", encoding_format="float")
            dim = len(resp.data[0].embedding)
            logger.info(f"Auto-detected dimension: {dim} for {model}")
            return dim
        except Exception as e:
            logger.error(f"Dimension detection failed: {e}")
            return self.settings.default_embedder_dim

    async def _auto_calibrate_if_new(self, model_key: str, vectors: List[List[float]]):
        try:
            pts = await self.backend.retrieve(
                collection_name=self._models_collection, ids=[model_key], 
                with_payload=["is_calibrated"]
            )
            if pts and pts[0].payload.get("is_calibrated"): return
        except Exception: pass

        if len(vectors) < 50: return

        def _compute():
            subset = random.sample(vectors, min(len(vectors), 100))
            norms = [math.sqrt(sum(x*x for x in v)) for v in subset]
            normed = [[x/n for x in v] for v, n in zip(subset, norms)]
            sims = []
            for i in range(len(normed)):
                for j in range(i + 1, len(normed)):
                    sims.append(sum(a * b for a, b in zip(normed[i], normed[j])))
            c_min, c_max = float(np.percentile(sims, 2)), float(np.percentile(sims, 98))
            if c_max - c_min < 0.1:
                c_min, c_max = max(-0.2, c_min - 0.05), min(0.95, c_max + 0.05)
            return round(c_min, 4), round(c_max, 4)

        c_min, c_max = await asyncio.to_thread(_compute)
        await self.backend.upsert(
            collection_name=self._models_collection, 
            points=[models.PointStruct(id=model_key, vector=[0.0], payload={
            "c_min": c_min, "c_max": c_max, "is_calibrated": True,
            "fitted_at": time.time(), "method": "batch_pairwise_percentiles", "sample_size": len(vectors)
        })])
        logger.info(f"Auto-calibrated {model_key}: c_min={c_min}, c_max={c_max}")

    async def _parse_to_markdown(self, req: EmbedRequest) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
        def _sync_process() -> tuple[List[Dict[str, Any]], Dict[str, str]]:
            all_chunks, documents_map = [], {}
            config = ExtractionConfig(output_format="markdown")
            sources = []
            if req.text: sources.append(("direct_input", req.record_id or f"text_{uuid.uuid4().hex[:8]}"))
            if req.url: sources.append((req.url, req.record_id or f"url_{uuid.uuid4().hex[:8]}"))
            if req.files_urls: sources.extend((u, req.record_id or f"file_{i}") for i, u in enumerate(req.files_urls))

            for src, doc_id in sources:
                try:
                    if src == "direct_input":
                        md_text = req.text.strip()
                        documents_map[doc_id] = "direct_input"
                    else:
                        loader = KreuzbergLoader(source=src, config=config)
                        docs = loader.load()
                        md_text = "\n\n".join(d.page_content for d in docs if d.page_content.strip())
                        documents_map[doc_id] = src
                    if md_text:
                        all_chunks.extend(cascade_chunk_markdown(md_text, doc_id, req.chunk_size, req.chunk_overlap, req.separator))
                except Exception as e:
                    logger.warning(f"Parse failed for {src}: {e}")
            return all_chunks, documents_map
        return await asyncio.to_thread(_sync_process)

    async def health(self) -> HealthResponse:
        try:
            await self.backend.get_collections()
            return HealthResponse(status="ok")
        except Exception:
            return HealthResponse(status="degraded")


    async def embed(self, req: EmbedRequest, tenant_ctx: dict, api_key: Optional[str], bg: BackgroundTasks) -> EmbedResponse:
        start = time.time()
        tenant_id, quota_kb = tenant_ctx["tenant_id"], tenant_ctx["quota_kb"]
        base_id = req.base_id or str(uuid.uuid4())

        # Конфигурация эмбеддера
        is_default = req.embedder_type == "platform" or not req.embedder_url
        actual_model = self.settings.default_embedder_model if is_default else req.embedder_model
        actual_url = self.settings.default_embedder_url if is_default else req.embedder_url
        actual_dim = self.settings.default_embedder_dim if is_default else req.embedder_dim
        collection_name = self._get_collection_name(actual_model)
        model_key = self._get_clean_model_key(actual_model)
        api_to_use = api_key or req.openai_api_key

        # 1. Логика перезаписи (если base_id уже существует в метаданных)
        if req.base_id:
            old_pts = await self.backend.retrieve(
                collection_name=self._meta_collection, ids=[req.base_id]
            )
            if old_pts:
                old_cfg = old_pts[0].payload
                old_coll = old_cfg.get("collection_name")
                await self._delete_base_points(old_coll, tenant_id, req.base_id)
                await self.backend.delete(            # Удаляем старый конфиг из метаданных
                    collection_name=self._meta_collection,
                    points_selector=models.PointIdsList(points=[req.base_id])
                )
                logger.info(f"♻️ Overwrite cleanup completed for base_id={req.base_id}")
            else:
                logger.warning(f"⚠️ Explicit base_id={req.base_id} not found during /embed")

        # 2. Парсинг и чанкование
        chunks, documents_map = await self._parse_to_markdown(req)
        if not chunks:
            raise HTTPException(400, detail="No valid chunks generated after parsing.")


        # 3. Динамический расчёт квоты (сумма size_kb по всем активным БЗ тенанта)
        points, _ = await self.backend.scroll(
            collection_name=self._meta_collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="type", match=models.MatchValue(value="kb_config")),
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
            ]),
            limit=10000, with_payload=["size_kb"], with_vectors=False
        )
        current_used_kb = sum(p.payload.get("size_kb", 0.0) for p in points)
        remaining_kb = quota_kb - current_used_kb

        
        # 4. Batch-эмбеддинг с проверкой квоты на лету
        texts = [c["text"] for c in chunks]
        all_vectors, total_tokens, total_cost, processed_kb = [], 0, 0.0, 0.0

        # Размер чанка с запасом под заголовки и кодировку
        chunk_size_kb = req.chunk_size * 1.4 / 1024.0  
        limit_kb = self.settings.embed_max_batch_kb - chunk_size_kb

        l = r = 0
        while l < len(texts):
            current_kb = 0.0

            while (r < len(texts) and current_kb < limit_kb) or (r==l):
                current_kb += len(texts[r].encode("utf-8")) / 1024.0
                r += 1

            batch_texts = texts[l:r]
            l = r

            if processed_kb + current_kb > remaining_kb:
                raise HTTPException(
                    413, detail=f"Storage quota exceeded. Available: ~{remaining_kb:.1f}KB"
                )

            vecs, t, c = await self._batch_embed(
                actual_model, actual_url, api_to_use, batch_texts, is_default
            )
            all_vectors.extend(vecs)
            total_tokens += t; total_cost += c; processed_kb += current_kb

        
        # 5. Автокалибровка (если модель новая)
        await self._auto_calibrate_if_new(model_key, all_vectors)

        # 6. Создание коллекции (если ещё нет)
        if not actual_dim:
            actual_dim = await self._detect_embedding_dimension(
                actual_model, actual_url, api_to_use
            )
        if not await self._collection_exists(collection_name):
            try:
                await self.backend.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=actual_dim, distance=models.Distance.COSINE
                    )
                )
                logger.info(f"🆕 Created collection {collection_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"ℹ️ Collection {collection_name} already exists (concurrent creation)")
                else:
                    raise HTTPException(500, detail=f"Collection init failed: {e}")

        # 7. Загрузка векторов в Qdrant
        points_to_upsert = [
            models.PointStruct(
                id=c["id"], vector=all_vectors[i],
                payload={
                    "tenant_id": tenant_id, "base_id": base_id, "text": c["text"],
                    "doc_id": c["doc_id"], "chunk_index": c["chunk_index"], "created_at": c["created_at"]
                }
            ) for i, c in enumerate(chunks)
        ]
        await self.backend.upsert(collection_name=collection_name, points=points_to_upsert)

        # 8. Сохранение конфига БЗ в метаданные (с точным размером)
        await self.backend.upsert(collection_name=self._meta_collection, points=[models.PointStruct(
            id=base_id, vector=[0.0], payload={
                "type": "kb_config", "tenant_id": tenant_id, "base_id": base_id,
                "collection_name": collection_name, "documents": documents_map,
                "embedding_config": {
                    "type": "platform" if is_default else "custom",
                    "model_name": actual_model, "dimension": actual_dim, "url": actual_url
                },
                "size_kb": round(processed_kb, 2), "created_at": time.time(), "updated_at": time.time()
            }
        )])

        # 9. Метрики и ответ
        EMBED_LATENCY.observe(time.time() - start)
        REQ_COUNTER.labels(endpoint="/embed", status="success").inc()
        return EmbedResponse(
            base_id=f"emb_db:{base_id}", chunks=len(chunks), tokens=total_tokens, cost_usd=total_cost
        )


    async def query(self, req: QueryRequest, tenant_ctx: dict, api_key: Optional[str]) -> QueryResponse:
        start = time.time()
        tenant_id = tenant_ctx["tenant_id"]
        base_ids = list({req.base_id} if req.base_id else set(req.base_ids or []))
        if not base_ids: raise HTTPException(400, detail="base_id or base_ids required")

        kb_configs = await self._fetch_kb_meta(tenant_id, base_ids)
        all_results, total_tokens, total_cost, model_keys_seen = [], 0, 0.0, set()

        for cfg in kb_configs.values():
            ec = cfg["embedding_config"]
            model_key = self._get_clean_model_key(ec['model_name'])
            model_keys_seen.add(model_key)
            
            cal_params = await self._get_model_calibration(model_key)
            vec, t, c = await self._batch_embed(ec["model_name"], ec.get("url"), api_key or req.openai_api_key, [req.question], ec.get("type") == "platform")
            total_tokens += t
            total_cost += c

            hits = await self.backend.search(collection_name=cfg["collection_name"], query_vector=vec[0], limit=req.k,
                query_filter=models.Filter(must=[
                    models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                    models.FieldCondition(key="base_id", match=models.MatchValue(value=cfg["base_id"]))
                ]), with_payload=["text", "doc_id"], with_vectors=False)
            
            scores, docs_map = [], cfg.get("documents", {})
            for h in hits:
                calibrated = self._apply_calibration(h.score, cal_params, model_key)
                scores.append(calibrated)
                if req.relevance_threshold is not None and calibrated < req.relevance_threshold: continue
                all_results.append(ChunkResult(base_id=f"emb_db:{cfg['base_id']}", id=str(h.id), page_content=h.payload.get("text", ""),
                    metadata=ChunkMetadata(doc_id=h.payload.get("doc_id", ""), source=docs_map.get(h.payload.get("doc_id"), "unknown")),
                    relevance=round(calibrated, 4), raw_distance=round(h.score, 4)))
            
            self._record_metrics(model_key, scores)
            await self._update_score_stats(model_key, scores)

        all_results.sort(key=lambda x: x.relevance, reverse=True)
        QUERY_LATENCY.labels(model_key=";".join(model_keys_seen)).observe(time.time() - start)
        REQ_COUNTER.labels(endpoint="/query", status="success").inc()
        return QueryResponse(results=all_results, tokens=total_tokens, cost_usd=total_cost)


    async def remove(self, req: RemoveRequest, tenant_ctx: dict) -> Dict[str, str]:
        tenant_id = tenant_ctx["tenant_id"]
        cfg = (await self._fetch_kb_meta(tenant_id, [req.base_id]))[req.base_id]
        
        # _delete_base_points сам решит: warning (нет коллекции) или raise (ошибка удаления)
        await self._delete_base_points(cfg["collection_name"], tenant_id, req.base_id)
        
        # Удаляем конфиг из мета-коллекции (квота освободится автоматически при следующем /embed)
        await self.backend.delete(
            collection_name=self._meta_collection, 
            points_selector=models.PointIdsList(points=[req.base_id])
        )
        
        REQ_COUNTER.labels(endpoint="/remove", status="success").inc()
        logger.info(f"🗑 Successfully removed base_id={req.base_id} for tenant {tenant_id}")
        return {"status": "success"}


# ─────────────────────────────────────────────────────────────────────────────
# 🌐 FastAPI App & Lifespan
# ─────────────────────────────────────────────────────────────────────────────
def get_kbs(request: Request) -> KBSService:
    return request.app.state.kbs

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings

    logger.info("KBS backend initialization")
    backend = AsyncQdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key, prefer_grpc=True
    )
    meta_cfg = models.HnswConfigDiff(
        m=1, payload_m=16, on_disk=True, memmap_threshold=1, full_scan_threshold=1
    )
    for name in [settings.meta_collection_name, settings.models_collection_name]:
        try:
            await backend.create_collection(
                collection_name=name, 
                vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT), 
                hnsw_config=meta_cfg
            )
            if name == settings.meta_collection_name:
                await backend.create_payload_index(
                    collection_name=name, field_name="tenant_id", field_schema="keyword"
                )
                await backend.create_payload_index(
                    collection_name=name, field_name="base_id", field_schema="keyword"
                )
            elif name == settings.models_collection_name:
                await backend.create_payload_index(
                    collection_name=name, field_name="model_name", field_schema="keyword"
                )
        except Exception as e:
            logger.warning(f"{name} init warning: {e}")

    app.state.kbs = KBSService(backend=backend, settings=settings)
    logger.info("KBS initialized successfully")
    yield
    logger.info("KBS shutdown...")
    await backend.close()
    logger.info("KBS shutdown complete")

app = FastAPI(title="Knowledge Base Service", version="1.0.0", lifespan=lifespan)
app.state.settings = KBSettings.from_env()

# Регистрация middleware для проксирования
if app.state.settings.legacy_query_url:
    app.add_middleware(
        LegacyProxyMiddleware,
        legacy_url=app.state.settings.legacy_query_url,
        timeout=app.state.settings.legacy_proxy_timeout
    )

# Регистрация middleware для сквозного логирования
app.add_middleware(RequestContextMiddleware)

@app.get("/api/v1/kb/health", response_model=HealthResponse)
async def health_check(kbs: KBSService = Depends(get_kbs)):
    return await kbs.health()

@app.post("/api/v1/kb/embed", response_model=EmbedResponse)
async def embed_kb(req: EmbedRequest, bg: BackgroundTasks, 
                   authorization: str = Header(..., alias="Authorization"), 
                   x_embed_key: Optional[str] = Header(None, alias="X-Embedding-API-Key"), 
                   kbs: KBSService = Depends(get_kbs)
):
    tenant_ctx = await kbs._extract_tenant_context(authorization)
    return await kbs.embed(req, tenant_ctx, x_embed_key, bg)

@app.post("/api/v1/kb/query", response_model=QueryResponse)
async def query_kb(req: QueryRequest, 
                   authorization: str = Header(..., alias="Authorization"), 
                   x_embed_key: Optional[str] = Header(None, alias="X-Embedding-API-Key"), 
                   kbs: KBSService = Depends(get_kbs)
):
    tenant_ctx = await kbs._extract_tenant_context(authorization)
    return await kbs.query(req, tenant_ctx, x_embed_key)

@app.post("/api/v1/kb/remove")
async def remove_kb(req: RemoveRequest, 
                    authorization: str = Header(..., alias="Authorization"), 
                    kbs: KBSService = Depends(get_kbs)
):
    tenant_ctx = await kbs._extract_tenant_context(authorization)
    return await kbs.remove(req, tenant_ctx)

@app.get("/metrics")
async def prometheus_metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="1.0.0.127", port=8000, 
                timeout_keep_alive=app.state.settings.uvicorn_timeout_keep_alive, 
                log_level=app.state.settings.log_level.lower()
    )
