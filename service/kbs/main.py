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
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import find_dotenv
from qdrant_client import AsyncQdrantClient, models
from openai import AsyncOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_kreuzberg import KreuzbergLoader
from kreuzberg import ExtractionConfig
from prometheus_client import Histogram, Counter, Gauge, REGISTRY, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
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
    default_api_key: Optional[str] = None

    jwt_secret: str
    jwt_algorithm: str = "HS256"

    default_tokens_price: float = 0.02
    embed_max_batch_kb: int = 64
    default_quota_kb: int = 5120

    default_chunk_size: int = 500
    default_chunk_overlap: int = 80
    default_separator: str = "--- BLOCK START ---"

    collection_prefix: str = "emb_"
    meta_collection_name: str = "kb_metadata"
    models_collection_name: str = "kb_models"

    calibration_default_min: float = 0.0
    calibration_default_max: float = 1.0
    calibration_min_delta:   float = 1e-6
    new_calibration_threshold: int = 50
    recalibration_window:      int = 500

    log_level: str = "INFO"
    uvicorn_timeout_keep_alive: int = 600

    legacy_query_url: Optional[str] = None
    legacy_proxy_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "KBSettings":
        return cls()

# ─────────────────────────────────────────────────────────────────────────────
# 📝 Логирование с контекстом (ТЗ 7.3)
# ─────────────────────────────────────────────────────────────────────────────
request_id_var = contextvars.ContextVar("request_id", default="-")
tenant_id_var = contextvars.ContextVar("tenant_id", default="-")
endpoint_var = contextvars.ContextVar("endpoint", default="-")

class ContextLogFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        record.tenant_id = tenant_id_var.get()
        record.endpoint = endpoint_var.get()
        # JSON-экранируем сообщение, чтобы оно было валидным внутри JSON-лога
        record.message_json = json.dumps(record.getMessage(), ensure_ascii=False)
        return True

logging.basicConfig(
    level=os.getenv("KBS_LOG_LEVEL", "INFO").upper(),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "service": "kbs", "request_id": "%(request_id)s", "tenant_id": "%(tenant_id)s", "endpoint": "%(endpoint)s", "msg": %(message_json)s}'
)
for handler in logging.root.handlers:
    handler.addFilter(ContextLogFilter())
logger = logging.getLogger("kbs")

# ─────────────────────────────────────────────────────────────────────────────
# 📊 Prometheus Metrics
# ─────────────────────────────────────────────────────────────────────────────
METRICS_PREFIX = "kbs_"
EMBED_LATENCY = Histogram(f"{METRICS_PREFIX}embed_duration_seconds", "Embedding pipeline latency")
QUERY_LATENCY = Histogram(f"{METRICS_PREFIX}query_duration_seconds", "Query pipeline latency", ["model_key"])
REQ_COUNTER = Counter(f"{METRICS_PREFIX}requests_total", "Total API requests", ["endpoint", "status"])
SCORE_DIST = Histogram(f"{METRICS_PREFIX}score_distribution", "Calibrated score distribution", ["model_key"], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

DRIFT_PSI_GAUGE = Gauge(f"{METRICS_PREFIX}drift_psi", "Population Stability Index vs previous baseline", ["model_key"])
SCORE_MEAN_GAUGE = Gauge(f"{METRICS_PREFIX}score_mean", "Rolling mean of normalized score", ["model_key"])

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
    embedder_url: Optional[str] = None
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
    raw_score: float = Field(..., description="Raw cosine score in range [-1, 1].")
    relevance: float = Field(..., description="Vector relevance=(1+cosine)/2 in range [0, 1].")

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
        separators=[separator, "\n\n", "\n", ". ", " ", "。"],
        keep_separator="end"
    )
    final_chunks = rec_splitter.split_documents(enriched)

    return [
        {"id": f"{doc_id}#chunk_{i}", "text": chunk.page_content, "doc_id": doc_id,
         "chunk_index": i, "created_at": time.time()}
        for i, chunk in enumerate(final_chunks) if chunk.page_content.strip()
    ]

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Stateless Service Class (DI + Lifecycle + Recalibration)
# ─────────────────────────────────────────────────────────────────────────────
class KBSService:
    def __init__(self, backend: AsyncQdrantClient, settings: KBSettings):
        self.backend = backend
        self.settings = settings
        self._meta_collection = settings.meta_collection_name
        self._models_collection = settings.models_collection_name

        # In-memory трекинг для rolling-калибровки
        self._score_windows: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=settings.recalibration_window))
        self._baselines: Dict[str, np.ndarray] = {}       # Кэш гистограммы
        self._calib_cache: Dict[str, dict] = {}           # Кэш c_min, c_max, version
        self._bucket_edges = np.linspace(0.0, 1.0, 51)    # 50 бинов
        self._lock = asyncio.Lock()

        self.out_of_range_counter = Counter(
            f"{METRICS_PREFIX}calibration_out_of_range_total",
            "Count of scores outside calibrated range", ["model_key"]
        )

    def _get_clean_model_key(self, model: str) -> str:
        base = model.split(":", 1)[0].strip()
        return base.replace("/", "_").replace(" ", "_").replace(".", "_")

    def _get_collection_name(self, model: str) -> str:
        return f"{self.settings.collection_prefix}{self._get_clean_model_key(model)}"

    async def _scroll_model(self, model_key: str, with_payload: bool = True) -> Optional[dict]:
        """Безопасный scroll по kb_models (gRPC-совместимый): возвращает точку или ее payload"""
        try:
            pts, _ = await self.backend.scroll(
                collection_name=self._models_collection,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="model_name", match=models.MatchValue(value=model_key))
                ]),
                limit=1, with_payload=with_payload, with_vectors=False
            )
            return pts if not with_payload else {"id": str(pts[0].id), **pts[0].payload} if pts else None
        except Exception as e:
            logger.error(f"❌ Error in {self._models_collection} scroll by {model_key}: {e}", exc_info=True)
            return None

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
            tenant_id_var.set(tenant_id)
            quota_kb = int(payload.get("tenant_quota_kb", self.settings.default_quota_kb))
            return {"tenant_id": tenant_id, "quota_kb": quota_kb}
        except (jwt.PyJWTError, ValueError, TypeError) as e:
            raise HTTPException(401, detail=f"Invalid token: {e}")

    async def _fetch_kb_meta(self, tenant_id: str, base_ids: List[str]) -> Dict[str, Dict]:
        ids = [id.split(':')[-1].strip() for id in base_ids]
        points, _ = await self.backend.scroll(
            collection_name=self._meta_collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                models.FieldCondition(key="base_id", match=models.MatchAny(any=ids))
            ]),
            limit=len(base_ids), with_payload=True, with_vectors=False
        )
        found = {p.payload["base_id"]: p.payload for p in points}
        missing = set(base_ids) - found.keys()
        if missing:
            raise HTTPException(403, detail=f"KB not found or access denied: {', '.join(missing)}")
        return found

    async def _delete_base_points(self, collection_name: str, tenant_id: str, base_id: str):
        if not collection_name:
            logger.error(f"❌ Internal error: collection_name is missing for base_id={base_id}. Aborting.")
            raise ValueError("Corrupted metadata: missing collection_name")

        selector = models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                models.FieldCondition(key="base_id", match=models.MatchValue(value=base_id))
            ])
        )
        try:
            await self.backend.delete(collection_name=collection_name, points_selector=selector)
            logger.info(f"✅ Deleted points for base_id={base_id} (tenant={tenant_id})")
        except Exception as e:
            is_missing = (getattr(e, "status_code", None) == 404) or \
                         (callable(getattr(e, "code", None)) and e.code() == grpc.StatusCode.NOT_FOUND)
            if is_missing:
                logger.warning(f"⚠️ Collection '{collection_name}' missing. Proceeding with metadata cleanup.")
                return
            logger.error(f"❌ Point deletion failed for base_id={base_id}: {e}")
            raise

    async def _get_model_calibration(self, model_key: str) -> Optional[Dict[str, float]]:
        """Возвращает кэшированные или загруженные границы калибровки в [0, 1]"""
        if model_key in self._calib_cache:
            cache = self._calib_cache[model_key]
            return {"min": cache["c_min"], "max": cache["c_max"]}

        state = await self._scroll_model(model_key)
        if state:
            self._calib_cache[model_key] = state
            if state.get("baseline_hist"):
                self._baselines[model_key] = np.array(state["baseline_hist"], dtype=np.float32)
            return {"min": state["c_min"], "max": state["c_max"]}
        return None


    def _apply_calibration(self, raw_cosine: float, cal_params: Optional[Dict], model_key: str = "unknown") -> float:
        # Преобразуем raw_cosine [-1, 1] в relevance [0, 1]
        relev = max(0.0, min(1.0, (raw_cosine + 1.0) / 2.0))
        if not cal_params:  return relev

        # Если модель калибрована - берем ее c_min и c_max
        c_min = cal_params.get("min", self.settings.calibration_default_min)
        c_max = cal_params.get("max", self.settings.calibration_default_max)
        if relev < c_min or relev > c_max:
            self.out_of_range_counter.labels(model_key=model_key).inc()

        # Нормализуем relevance в калибровочный интервал
        if c_max - c_min < self.settings.calibration_min_delta:  return relev
        return max(0.0, min(1.0, (relev - c_min) / (c_max - c_min)))


    def _calc_psi(self, baseline: np.ndarray, current: np.ndarray) -> float:
        s_base = max(baseline.sum(), 1e-6)
        s_curr = max(current.sum(), 1e-6)
        psi = 0.0
        for b, c in zip(baseline, current):
            pct_b = b / s_base
            pct_c = c / s_curr
            if pct_b != 0 or pct_c != 0:
                psi += (pct_c - pct_b) * math.log((pct_c + 1e-6) / (pct_b + 1e-6))
        return  psi


    def _record_drift_metrics(self, model_key: str, curr_hist: np.ndarray, base_hist: np.ndarray, window: deque):
        psi = self._calc_psi(base_hist, curr_hist) if base_hist.sum() > 0 else 0.0
        DRIFT_PSI_GAUGE.labels(model_key=model_key).set(round(psi, 4))
        SCORE_MEAN_GAUGE.labels(model_key=model_key).set(round(np.mean(list(window)), 4) if window else 0.0)
        

    async def _update_score_stats(self, model_key: str, raw_scores: List[float]):
        """Rolling-калибровка + baseline + PSI (работает даже при cold-start <50)"""
        async with self._lock:
            # Преобразуем raw_scores в relevance и отписываем в скользящее окно
            self._score_windows[model_key].extend((np.array(raw_scores, dtype=np.float32) + 1)/2)

            # Делаем snapshot окна и выходим, если не набрали 500 скоров
            window = list(self._score_windows[model_key])
            if len(window) < self.settings.recalibration_window: return

        # 1. Загружаем текущее состояние из Qdrant (или None, если холодный старт)
        state = await self._scroll_model(model_key) or {}
        baseline_hist = np.array(state.get("baseline_hist", []), dtype=np.float32)
        old_c_min = state.get("c_min", self.settings.calibration_default_min)
        old_c_max = state.get("c_max", self.settings.calibration_default_max)
        version = state.get("calib_version", 0)

        # 2. Если baseline ещё нет, создаём его из текущего окна, иначе сливаем гистограммы
        wind_hist, _ = np.histogram(list(window), bins=self._bucket_edges, density=False)
        if baseline_hist.size == 0:
            baseline_hist = merged_hist = wind_hist
            logger.info(f"📊 Created initial baseline for {model_key} from first {len(window)} scores")
        else:
            merged_hist = (baseline_hist + wind_hist).astype(np.float32)
        
        # Отписываем метрики дрифта статистик relevance
        self._record_drift_metrics(model_key, wind_hist, baseline_hist, window)
        
        # 3. Считаем новые границы калибровочного диапазона по 2- и 98-персентилям
        cdf = np.cumsum(merged_hist)
        total = cdf[-1]
        idx_min = np.clip(np.searchsorted(cdf, total * 0.02), 0, len(self._bucket_edges)-1)
        idx_max = np.clip(np.searchsorted(cdf, total * 0.98), 0, len(self._bucket_edges)-1)
        new_c_min = round(self._bucket_edges[idx_min], 4)
        new_c_max = round(self._bucket_edges[idx_max], 4)

        # 4. Не обновляем калибровку если границы не сместились
        if abs(new_c_min - old_c_min) < 0.02 and abs(new_c_max - old_c_max) < 0.02:
            logger.debug(f"Drift within tolerance for {model_key}, skipping recalibration...")
            async with self._lock:
                self._score_windows[model_key].clear()
            return

        async with self._lock:
            # 5. OCC-запись в Qdrant (безопасно при конкурентности)
            latest = await self._scroll_model(model_key) or {}
            if latest.get("calib_version", 0) != version:
                logger.debug(f"⏭️ Calibration race for {model_key}, skipping recalibration...")
                # OCC: если версия изменилась, значит другой процесс уже обновил калибровку
                # и очистил окно. Мы выходим без побочных эффектов, чтобы не потерять
                # новые скоры, накопившиеся после очистки победителем.
                #window.clear()
                return

            await self.backend.upsert(
                collection_name=self._models_collection,
                points=[models.PointStruct(
                    id=latest.get('id', str(uuid.uuid4())), 
                    vector=[0.0],
                    payload={
                        "model_name": model_key,
                        "c_min": new_c_min, "c_max": new_c_max,
                        "calib_version": version + 1, 
                        "total_scores": latest.get("total_scores", 0) + len(window),
                        "baseline_hist": merged_hist.tolist(),
                        "updated_at": time.time()
                    }
                )]
            )

            # 6. Обновляем in-memory кэш и логи
            self._baselines[model_key] = merged_hist
            self._calib_cache[model_key] = {"c_min": new_c_min, "c_max": new_c_max}
            
            self._score_windows[model_key].clear()
        logger.info(f"✅ Recalibrated {model_key}: [{new_c_min}, {new_c_max}] v{version+1}")


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
        """Первичная калибровка модели"""
        state = await self._scroll_model(model_key, with_payload=False)
        if state or len(vectors) < self.settings.new_calibration_threshold:  return

        def _compute():
            subset = np.array(random.sample(vectors, min(len(vectors), 100)), dtype=np.float32)
            norms  = np.linalg.norm(subset, axis=1, keepdims=True)
            norms[norms<1e-9] = 1e-9
            normed = subset / norms
            ij = np.triu_indices(len(normed), k=1)
            sims = ((normed @ normed.T)[ij] +1) /2
            c_min, c_max = np.percentile(sims, 2), np.percentile(sims, 98)
            hist, _ = np.histogram(sims, bins=self._bucket_edges, density=False)
            return round(c_min, 4), round(c_max, 4), hist.astype(np.float32)

        c_min, c_max, baseline = await asyncio.to_thread(_compute)
        
        async with self._lock:
            if  self._scroll_model(model_key, with_payload=False):
                logger.debug(f"⏭️ Calibration race for {model_key}, skipping initial calibration...")
                return
            await self.backend.upsert(
                collection_name=self._models_collection,
                points=[models.PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=[0.0],
                    payload={
                        "model_name": model_key,
                        "c_min": c_min, "c_max": c_max,
                        "calib_version": 1, "total_scores": 0,
                        "baseline_hist": baseline.tolist(),
                        "updated_at": time.time()
                    }
                )]
            )
            self._calib_cache[model_key] = {"c_min": c_min, "c_max": c_max, "calib_version": 1}
            self._baselines[model_key] = baseline
        logger.info(f"🆕 Initial calibration for {model_key}: [{c_min}, {c_max}]")

    async def _parse_and_chunk(self, req: EmbedRequest) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
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


    async def _collection_exists(self, collection_name: str) -> bool:
        try:
            collections = await self.backend.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            logger.error(f"❌ Qdrant.get_collections() failed: {e}")
            return False


    async def _create_collection(self, collection_name: str, actual_dim: int) -> None:
        try:
            await self.backend.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=actual_dim, distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(payload_m=16,m=0)
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.error(msg:=f"Collection `{collection_name}` init failed: {e}", exc_info=True)
                raise HTTPException(500, detail=msg)
        try:
            await self.backend.create_payload_index(
                collection_name=collection_name,
                field_name="tenant_id",
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
            ))
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.error(msg:=f"Index init by tenant_id for `{collection_name}` failed: {e}", exc_info=True)
                raise HTTPException(500, detail=msg)
        try:
            await self.backend.create_payload_index(
                collection_name=collection_name,
                field_name="base_id",
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
            ))
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.error(msg:=f"Index init by base_id for `{collection_name}` failed: {e}", exc_info=True)
                raise HTTPException(500, detail=msg)
        logger.info(f"🆕 Created collection `{collection_name}`")


    async def health(self) -> HealthResponse:
        try:
            await self.backend.get_collections()
            return HealthResponse(status="ok")
        except Exception:
            return HealthResponse(status="degraded")


    async def embed(self, req: EmbedRequest, tenant_ctx: dict, api_key: Optional[str], bg: BackgroundTasks) -> EmbedResponse:
        start = time.time()
        tenant_id, quota_kb = tenant_ctx["tenant_id"], tenant_ctx["quota_kb"]
        base_id = req.base_id.split(':')[-1].strip() or str(uuid.uuid4())
        filter  = models.Filter(must=[
            models.FieldCondition(key="base_id", match=models.MatchValue(value=base_id)),
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
        ])

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
            old_pts, _ = await self.backend.scroll(
                collection_name=self._meta_collection,
                scroll_filter=filter,
                limit=1, with_payload=["collection_name"], with_vectors=False
            )
            if old_pts:
                old_coll = old_pts[0].payload.get("collection_name")
                await self._delete_base_points(old_coll, tenant_id, req.base_id)
                await self.backend.delete(  # Удаляем старый конфиг из метаданных (осв. квоту!)
                    collection_name=self._meta_collection,
                    points_selector=models.PointIdsList(points=[old_pts[0].id])
                )
                logger.info(f"♻️ Overwrite cleanup completed for base_id={req.base_id}")
            else:
                logger.warning(f"⚠️ Explicit base_id={req.base_id} not found during /embed")

        # 2. Парсинг и чанкование
        chunks, documents_map = await self._parse_and_chunk(req)
        if not chunks:
            raise HTTPException(400, detail="No valid chunks generated after parsing.")


        # 3. Динамический расчёт квоты (сумма size_kb по всем активным БЗ тенанта)
        points, _ = await self.backend.scroll(
            collection_name=self._meta_collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="type", match=models.MatchValue(value="kb_config")),
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
            ]),
            #limit=10000, 
            with_payload=["size_kb"], with_vectors=False
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

            while r < len(texts) and (current_kb < limit_kb or r==l):
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
        async with self._lock:
            if not await self._collection_exists(collection_name):
                await self._create_collection(collection_name, actual_dim)

        # 7. Write-ahead конфига БЗ в метаданные
        await self.backend.upsert(collection_name=self._meta_collection, points=[models.PointStruct(
            id=str(uuid.uuid4()), 
            vector=[0.0], payload={
                "type": "kb_config", "tenant_id": tenant_id, "base_id": base_id,
                "collection_name": collection_name, "documents": documents_map,
                "embedding_config": {
                    "type": "platform" if is_default else "custom",
                    "model_name": actual_model, "dimension": actual_dim, "url": actual_url
                },
                "size_kb": round(processed_kb, 2), "created_at": time.time()
            }
        )])
        logger.info(f"✅ Metadata write-ahead completed for base_id={base_id}")

        # 8. Загрузка векторов в Qdrant
        points_to_upsert = [
            models.PointStruct(
                id=str(uuid.uuid4()), 
                vector=all_vectors[i],
                payload={
                    "tenant_id": tenant_id, "base_id": base_id, "text": c["text"],
                    "doc_id": c["doc_id"], "chunk_index": c["chunk_index"], "created_at": c["created_at"]
                }
            ) for i, c in enumerate(chunks)
        ]
        try:
            await self.backend.upsert(collection_name=collection_name, points=points_to_upsert)
            logger.info(f"✅ Data upload completed for base_id={base_id} ({len(points_to_upsert)} points)")
        except Exception as e:
            logger.error(f"❌ Vector upsert failed for base_id={base_id}: {e}", exc_info=True)
            raise HTTPException(500, detail=f"Failed to store vectors: {e}")

        # 9. Отражаем факт успешной загрузки в конфиге БЗ
        await self.backend.set_payload(
            collection_name=self._meta_collection,
            payload={"updated_at": time.time()},
            points=filter
        )
        logger.info(f"✅ Metadata finalized for base_id={base_id}")

        # 10. Метрики и ответ
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

        # Извлекаем сведения о требуемых БЗ из kb_metadata
        kb_configs = await self._fetch_kb_meta(tenant_id, base_ids)
        all_results, total_tokens, total_cost, model_keys_seen = [], 0, 0.0, set()

        # Извлекаем релевантные запросу чанки из каждой БЗ
        for cfg in kb_configs.values():
            if  not "updated_at" in cfg: raise HTTPException(
                423,  # Locked
                detail=f"Knowledge base {cfg['base_id']} is still loading or load has been failed."
            )

            # Извлекаем конфигурацию эмбеддера для данной БЗ
            ec = cfg["embedding_config"]
            model_key = self._get_clean_model_key(ec['model_name'])
            model_keys_seen.add(model_key)
            cal_params = await self._get_model_calibration(model_key)

            # Получаем эмбеддинг запроса пользователя
            vec, t, c = await self._batch_embed(
                ec["model_name"], ec.get("url"), api_key or req.openai_api_key, 
                [req.question], ec.get("type") == "platform"
            )
            total_tokens += t; total_cost += c

            # Ищем релевантные чанки в БЗ (количество чанков берем с запасом)
            hits = await self.backend.query_points(collection_name=cfg["collection_name"], 
                query=vec[0], limit=req.k * len(kb_configs),
                query_filter=models.Filter(must=[
                    models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                    models.FieldCondition(key="base_id", match=models.MatchValue(value=cfg["base_id"]))
                ]), with_payload=["text", "doc_id", "chunk_index"], with_vectors=False)
            logger.debug(f"✅ Retrieved points: {hits}")
            
            # Фильтруем извлеченные чанки по relevance_threshold и добавляем их в all_results
            raw_scores, scores, docs_map = [], [], cfg.get("documents", {})
            for h in hits.points:
                raw_scores.append(h.score)
                calibrated = self._apply_calibration(h.score, cal_params, model_key)
                scores.append(calibrated)
                if req.relevance_threshold and calibrated < req.relevance_threshold: continue
                all_results.append(ChunkResult(
                    base_id=f"emb_db:{cfg['base_id']}", 
                    id=str(h.id), page_content=h.payload.get("text", ""),
                    metadata=ChunkMetadata(
                        doc_id= h.payload.get('doc_id', ""), 
                        source= docs_map.get(h.payload.get('doc_id'), "unknown") + 
                                f": chunk {h.payload.get('chunk_index', 0)}"
                    ),
                    relevance=round(calibrated, 4), raw_score=round(h.score, 4)
                ))
            
            # Отписываем scores в метрики 
            for s in scores: SCORE_DIST.labels(model_key=model_key).observe(s)
            await self._update_score_stats(model_key, raw_scores)

        # Сортируем все извлеченные чанки по релевантности и формируем ответ
        all_results.sort(key=lambda x: x.relevance, reverse=True)
        QUERY_LATENCY.labels(model_key="; ".join(model_keys_seen)).observe(time.time() - start)
        REQ_COUNTER.labels(endpoint="/query", status="success").inc()
        return QueryResponse(results=all_results[:req.k], tokens=total_tokens, cost_usd=total_cost)


    async def remove(self, req: RemoveRequest, tenant_ctx: dict) -> Dict[str, str]:
        tenant_id = tenant_ctx['tenant_id']
        kb_configs = await self._fetch_kb_meta(tenant_id, [req.base_id])
        
        # Удаляем точки БЗ и ее конфиг из мета-коллекции
        for cfg in kb_configs.values():
            await self._delete_base_points(cfg['collection_name'], tenant_id, cfg['base_id'])
            await self.backend.delete(
                collection_name=self._meta_collection, 
                points_selector=models.Filter(must=[
                    models.FieldCondition(key="base_id", match=models.MatchValue(value=cfg['base_id'])),
                    models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
                ])
            )
            logger.info(f"🗑 Successfully removed base_id={cfg['base_id']} for tenant {tenant_id}")
        
        if  kb_configs:
            REQ_COUNTER.labels(endpoint="/remove", status="success").inc()
        else:
            logger.warning(f"⚠️ Not found base_id={req.base_id} for tenant {tenant_id}")
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
        m=0, payload_m=16, on_disk=True, full_scan_threshold=1
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
                    collection_name=name, field_name="tenant_id", 
                    field_schema=models.KeywordIndexParams(
                        type=models.KeywordIndexType.KEYWORD,
                        is_tenant=True,
                    )
                )
                await backend.create_payload_index(
                    collection_name=name, field_name="base_id",
                    field_schema=models.KeywordIndexParams(
                        type=models.KeywordIndexType.KEYWORD,
                    )
                )
            elif name == settings.models_collection_name:
                await backend.create_payload_index(
                    collection_name=name, field_name="model_name",
                    field_schema=models.KeywordIndexParams(
                        type=models.KeywordIndexType.KEYWORD,
                    )
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

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Middleware для сквозного логирования (ТЗ 7.3)
# ─────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_var.set(req_id)
    endpoint_var.set(f"{request.method} {request.url.path}")
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Middleware для проксирования запросов на legacy endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def legacy_proxy_middleware(request: Request, call_next):
    """
    Проксирует запросы к старым БЗ на legacy-эндпоинт.
    Если base_id найден в kb_metadata → обработка в новом сервисе.
    Если base_id отсутствует → запрос проксируется на legacy_url.
    """
    settings = request.app.state.settings

    # Работаем только с POST /api/v1/kb/query и если legacy_url задан
    if request.method != "POST" or request.url.path != "/api/v1/kb/query" or not settings.legacy_query_url:
        return await call_next(request)

    body_bytes = await request.body()

    try:
        payload = json.loads(body_bytes)
        target_id = payload.get("base_id") or (payload.get("base_ids") or [None])[0]

        if target_id:
            backend = request.app.state.kbs.backend
            found, _ = await backend.scroll(
                collection_name=settings.meta_collection_name,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="base_id", match=models.MatchValue(value=target_id))
                ]),
                limit=1, with_payload=False, with_vectors=False
            )
            if not found:
                # Базы нет в Qdrant → проксируем на legacy
                headers = {k: v for k, v in request.headers.items()
                           if k.lower() not in ("host", "content-length", "transfer-encoding")}
                async with httpx.AsyncClient(timeout=settings.legacy_proxy_timeout) as client:
                    resp = await client.post(settings.legacy_query_url, content=body_bytes, headers=headers)
                return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
    except Exception:
        pass  # Fail-open: при любой ошибке идём в новый сервис

    # Восстанавливаем body для call_next() (стандартный паттерн Starlette)
    async def receive():
        return {"type": "http.request", "body": body_bytes}
    request._receive = receive
    return await call_next(request)



# ─────────────────────────────────────────────────────────────────────────────
#  Эндпоинты микросервиса
# ─────────────────────────────────────────────────────────────────────────────
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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, 
                timeout_keep_alive=app.state.settings.uvicorn_timeout_keep_alive, 
                log_level=app.state.settings.log_level.lower()
    )
