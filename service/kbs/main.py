# main.py
"""
Knowledge Base Service (KBS) v1.0.0
Микросервис управления базами знаний на FastAPI + Qdrant
"""
import os
import time
import asyncio
import logging
import jwt
import uuid
import json
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, Depends, Header, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models
from openai import AsyncOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from prometheus_client import Histogram, Counter, Gauge, REGISTRY, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# ─────────────────────────────────────────────────────────────────────────────
# 🔧 Logging & Prometheus Metrics
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","service":"kbs","msg":"%(message)s"}'
)
logger = logging.getLogger("kbs")

METRICS_PREFIX = "kbs_"
EMBED_LATENCY = Histogram(f"{METRICS_PREFIX}embed_duration_seconds", "Embedding pipeline latency")
QUERY_LATENCY = Histogram(f"{METRICS_PREFIX}query_duration_seconds", "Query pipeline latency", ["model_key"])
REQ_COUNTER = Counter(f"{METRICS_PREFIX}requests_total", "Total API requests", ["endpoint", "status"])
SCORE_DIST = Histogram(f"{METRICS_PREFIX}score_distribution", "Calibrated score distribution", ["model_key"], buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ECE_GAUGE = Gauge(f"{METRICS_PREFIX}ece_per_model", "Expected Calibration Error per model", ["model_key"])

# ─────────────────────────────────────────────────────────────────────────────
# 📦 Pydantic Models (строго по ТЗ раздел 5.3)
# ─────────────────────────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    openai_api_key: Optional[str] = None
    base_id: Optional[str] = None
    prefix: str = "kb_"
    chunk_size: int = 500  # По умолчанию 500 символов (согласовано)
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
    base_id: Optional[str] = None
    base_ids: Optional[List[str]] = None
    question: str = Field(..., min_length=1)
    k: int = Field(3, ge=1, le=50)
    relevance_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

class ChunkMetadata(BaseModel):
    doc_id: str
    source: str
    page: str = "-"  # Для совместимости

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
    uptime: str = "running - для совместимости"
    version: str = "1.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# 🧩 Каскадный чанкер (MarkdownHeaderTextSplitter -> RecursiveCharacterTextSplitter)
# ─────────────────────────────────────────────────────────────────────────────
def cascade_chunk_markdown(
    markdown_text: str,
    doc_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    separator: str = "--- BLOCK START ---"
) -> List[Dict[str, Any]]:
    """
    Каскадное чанкование с инъекцией заголовков.
    chunk_size/overlap считаются в символах (len()).
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")],
        strip_headers=False
    )
    header_docs = md_splitter.split_text(markdown_text)

    enriched = []
    for doc in header_docs:
        # Формируем путь заголовков: H1 > H2 > H3
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
        {
            "id": f"{doc_id}#chunk_{i}",
            "text": chunk.page_content,
            "doc_id": doc_id,
            "chunk_index": i,
            "created_at": time.time()
        }
        for i, chunk in enumerate(final_chunks)
    ]

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Stateless Service Class (DI + Lifecycle)
# ─────────────────────────────────────────────────────────────────────────────
class KBSService:
    def __init__(self, backend: AsyncQdrantClient, default_api_key: str, jwt_secret: str):
        self.backend = backend
        self.default_api_key = default_api_key
        self.jwt_secret = jwt_secret
        self._pricing = {"text-embedding-3-small": 0.02, "text-embedding-ada-002": 0.10}
        self._meta_collection = "kb_metadata"

    async def _extract_tenant_id(self, auth_header: Optional[str]) -> str:
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(401, detail="Missing or invalid Bearer token")
        try:
            payload = jwt.decode(
                auth_header.split(" ", 1)[1], self.jwt_secret, algorithms=["HS256"],
                options={"verify_exp": True, "verify_iss": True, "verify_aud": True}
            )
            tenant_id = payload.get("tenant_id") or payload.get("sub")
            if not tenant_id: raise ValueError("tenant_id claim missing")
            return tenant_id
        except jwt.PyJWTError as e:
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

    def _apply_calibration(self, raw_cosine: float, cal_params: Optional[Dict]) -> float:
        """Нормализация (1+cos)/2 + калибровочный маппинг"""
        local = (raw_cosine + 1.0) / 2.0
        if not cal_params:
            return max(0.0, min(1.0, local))
        c_min, c_max = cal_params.get("min", 0.0), cal_params.get("max", 1.0)
        if c_max - c_min < 1e-6: return 0.5
        return max(0.0, min(1.0, (local - c_min) / (c_max - c_min)))

    def _record_metrics(self, model_key: str, scores: List[float]):
        for s in scores:
            SCORE_DIST.labels(model_key=model_key).observe(s)
        # ECE вычисляется асинхронно на основе фидбэка. Заглушка для алертинга.
        ECE_GAUGE.labels(model_key=model_key).set(0.0)

    async def _batch_embed(self, model: str, url: str, api_key: str, texts: List[str], is_default: bool):
        """Batch-эмбеддинг для ускорения /embed. Возвращает (vectors, tokens, cost)"""
        if is_default:
            client = AsyncOpenAI(api_key=self.default_api_key, base_url=url or "https://api.openai.com/v1")
            # OpenAI API поддерживает до 2048 текстов за вызов
            resp = await client.embeddings.create(model=model, input=texts, encoding_format="float")
            tokens = resp.usage.total_tokens
            cost = round((tokens / 1_000_000) * self._pricing.get(model, 0.02), 6)
            return [d.embedding for d in resp.data], tokens, cost
        else:
            # Custom: платформа не биллит
            embedder = AsyncOpenAI(api_key=api_key, base_url=url or "https://api.openai.com/v1")
            resp = await embedder.embeddings.create(model=model, input=texts, encoding_format="float")
            return [d.embedding for d in resp.data], 0, 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # 🌐 Endpoint Handlers
    # ─────────────────────────────────────────────────────────────────────────
    async def health(self) -> HealthResponse:
        try:
            await self.backend.get_collections()
            return HealthResponse(status="ok")
        except Exception:
            return HealthResponse(status="degraded")

    async def embed(self, req: EmbedRequest, tenant_id: str, api_key: Optional[str], bg: BackgroundTasks) -> EmbedResponse:
        start = time.time()
        base_id = req.base_id or str(uuid.uuid4())
        doc_id = req.record_id or f"doc-{uuid.uuid4().hex[:8]}"
        collection_name = f"kb_{tenant_id}_{base_id}"
        model_key = f"{req.embedder_url or 'openai'}:{req.embedder_model}"
        is_default = req.embedder_type == "platform"
        api_to_use = api_key or req.openai_api_key

        # 1. Парсинг в Markdown (CPU-bound -> thread)
        async def _parse_sync():
            if req.text: return req.text
            if req.url: return extract_markdown(req.url)
            if req.files_urls: return "\n\n".join(extract_markdown(u) for u in req.files_urls if u)
            return ""
        md_text = await asyncio.to_thread(_parse_sync)
        if not md_text.strip():
            raise HTTPException(400, detail="No content provided.")

        # 2. Каскадное чанкование
        chunks = cascade_chunk_markdown(md_text, doc_id, req.chunk_size, req.chunk_overlap, req.separator)
        if not chunks:
            raise HTTPException(400, detail="Failed to generate chunks.")

        # 3. Создание коллекции (если новая)
        try:
            collections = await self.backend.get_collections()
            exists = any(c.name == collection_name for c in collections.collections)
            if not exists:
                await self.backend.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=req.embedder_dim, distance=models.Distance.COSINE)
                )
        except Exception as e:
            raise HTTPException(500, detail=f"Qdrant collection init failed: {e}")

        # 4. Batch-векторизация
        texts = [c["text"] for c in chunks]
        # Разбиваем на батчи по 128 для стабильности
        batch_size = 128
        all_vectors = []
        total_tokens, total_cost = 0, 0.0
        for i in range(0, len(texts), batch_size):
            vecs, t, c = await self._batch_embed(req.embedder_model, req.embedder_url, api_to_use, texts[i:i+batch_size], is_default)
            all_vectors.extend(vecs)
            total_tokens += t
            total_cost += c

        # 5. Upsert в Qdrant
        points = [
            models.PointStruct(id=c["id"], vector=all_vectors[i], payload={
                "tenant_id": tenant_id, "base_id": base_id,
                "text": c["text"], "doc_id": c["doc_id"],
                "chunk_index": c["chunk_index"], "created_at": c["created_at"]
            }) for i, c in enumerate(chunks)
        ]
        await self.backend.upsert(collection_name=collection_name, points=points)

        # 6. Мета-коллекция: создание/обновление
        cal_params = {"min": 0.0, "max": 1.0, "fitted_at": time.time()}
        meta_point = models.PointStruct(id=base_id, vector=[0.0], payload={
            "type": "kb_config", "tenant_id": tenant_id, "base_id": base_id,
            "collection_name": collection_name,
            "documents": {doc_id: "uploaded_doc"},
            "embedding_config": {
                "type": "platform" if is_default else "custom",
                "model_name": req.embedder_model, "dimension": req.embedder_dim,
                "url": req.embedder_url, "calibration_params": cal_params
            },
            "disk_usage": 0, "created_at": time.time(), "updated_at": time.time()
        })
        await self.backend.upsert(collection_name=self._meta_collection, points=[meta_point])

        EMBED_LATENCY.observe(time.time() - start)
        REQ_COUNTER.labels(endpoint="/embed", status="success").inc()
        return EmbedResponse(base_id=f"emb_db:{base_id}", chunks=len(chunks), tokens=total_tokens, cost_usd=total_cost)

    async def query(self, req: QueryRequest, tenant_id: str, api_key: Optional[str]) -> QueryResponse:
        start = time.time()
        base_ids = list({req.base_id} if req.base_id else set(req.base_ids or []))
        if not base_ids: raise HTTPException(400, detail="base_id or base_ids required")

        kb_configs = await self._fetch_kb_meta(tenant_id, base_ids)
        all_results, total_tokens, total_cost = [], 0, 0.0
        model_keys_seen = set()

        for cfg in kb_configs.values():
            ec = cfg["embedding_config"]
            model_key = f"{ec.get('url', 'openai')}:{ec['model_name']}"
            model_keys_seen.add(model_key)
            
            # 1. Embed query
            vec, t, c = await self._batch_embed(
                ec["model_name"], ec.get("url"), api_key or req.openai_api_key,
                [req.question], req.embedder_type == "platform"
            )
            total_tokens += t
            total_cost += c

            # 2. Search Qdrant (строгая изоляция tenant_id + base_id)
            hits = await self.backend.search(
                collection_name=cfg["collection_name"], query_vector=vec[0], limit=req.k,
                query_filter=models.Filter(must=[
                    models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                    models.FieldCondition(key="base_id", match=models.MatchValue(value=cfg["base_id"]))
                ]),
                with_payload=["text", "doc_id"], with_vectors=False
            )
            
            cal_params = cfg.get("embedding_config", {}).get("calibration_params", {})
            scores = []
            docs_map = cfg.get("documents", {})
            for h in hits:
                calibrated = self._apply_calibration(h.score, cal_params)
                scores.append(calibrated)
                
                # Фильтрация по порогу: если None - пропускаем все
                if req.relevance_threshold is not None and calibrated < req.relevance_threshold:
                    continue
                    
                all_results.append(ChunkResult(
                    base_id=f"emb_db:{cfg['base_id']}", id=str(h.id),
                    page_content=h.payload.get("text", ""),
                    metadata=ChunkMetadata(
                        doc_id=h.payload.get("doc_id", ""),
                        source=docs_map.get(h.payload.get("doc_id"), "unknown")
                    ),
                    relevance=round(calibrated, 4),
                    raw_distance=round(h.score, 4)
                ))
            self._record_metrics(model_key, scores)

        # Сортировка по калиброванной релевантности
        all_results.sort(key=lambda x: x.relevance, reverse=True)
        
        QUERY_LATENCY.labels(model_key=";".join(model_keys_seen)).observe(time.time() - start)
        REQ_COUNTER.labels(endpoint="/query", status="success").inc()
        return QueryResponse(results=all_results, tokens=total_tokens, cost_usd=total_cost)

    async def remove(self, req: RemoveRequest, tenant_id: str) -> Dict[str, str]:
        cfg = (await self._fetch_kb_meta(tenant_id, [req.base_id]))[req.base_id]
        await self.backend.delete_collection(cfg["collection_name"])
        await self.backend.delete(
            collection_name=self._meta_collection,
            points_selector=models.PointIdsList(points=[req.base_id])
        )
        REQ_COUNTER.labels(endpoint="/remove", status="success").inc()
        return {"status": "success"}

# ─────────────────────────────────────────────────────────────────────────────
# 🌐 FastAPI App & Lifespan (Zero Globals)
# ─────────────────────────────────────────────────────────────────────────────
def get_kbs(request: Request) -> KBSService:
    return request.app.state.kbs

@asynccontextmanager
async def lifespan(app: FastAPI):
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    qdrant_api = os.getenv("QDRANT_API_KEY")
    backend = AsyncQdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api if qdrant_api else None,
        prefer_grpc=True
    )
    
    # Создаем мета-коллекцию при старте (если отсутствует)
    try:
        await backend.create_collection(
            collection_name="kb_metadata",
            vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
            hnsw_config=models.HnswConfigDiff(
                m=1, payload_m=16, 
                on_disk=True, memmap_threshold=1, full_scan_threshold=1
            )
        )
        # Индексы для быстрой фильтрации
        await backend.create_payload_index(collection_name="kb_metadata", field_name="tenant_id", field_schema="keyword")
        await backend.create_payload_index(collection_name="kb_metadata", field_name="base_id", field_schema="keyword")
    except Exception as e:
        logger.warning(f"Metadata collection init warning (might already exist): {e}")

    app.state.kbs = KBSService(
        backend=backend,
        default_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        jwt_secret=os.getenv("JWT_SECRET", "dev-secret-change-me")
    )
    logger.info("KBS initialized successfully")
    yield
    await backend.close()
    logger.info("KBS shutdown complete")

app = FastAPI(title="Knowledge Base Service", version="1.0.0", lifespan=lifespan)

# ─────────────────────────────────────────────────────────────────────────────
# 🌍 Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/kb/health", response_model=HealthResponse)
async def health_check(kbs: KBSService = Depends(get_kbs)):
    return await kbs.health()

@app.post("/api/v1/kb/embed", response_model=EmbedResponse)
async def embed_kb(
    req: EmbedRequest, bg: BackgroundTasks,
    authorization: str = Header(..., alias="Authorization"),
    x_embed_key: Optional[str] = Header(None, alias="X-Embedding-API-Key"),
    kbs: KBSService = Depends(get_kbs)
):
    tenant_id = await kbs._extract_tenant_id(authorization)
    return await kbs.embed(req, tenant_id, x_embed_key, bg)

@app.post("/api/v1/kb/query", response_model=QueryResponse)
async def query_kb(
    req: QueryRequest,
    authorization: str = Header(..., alias="Authorization"),
    x_embed_key: Optional[str] = Header(None, alias="X-Embedding-API-Key"),
    kbs: KBSService = Depends(get_kbs)
):
    tenant_id = await kbs._extract_tenant_id(authorization)
    return await kbs.query(req, tenant_id, x_embed_key)

@app.post("/api/v1/kb/remove")
async def remove_kb(
    req: RemoveRequest,
    authorization: str = Header(..., alias="Authorization"),
    kbs: KBSService = Depends(get_kbs)
):
    tenant_id = await kbs._extract_tenant_id(authorization)
    return await kbs.remove(req, tenant_id)

@app.get("/metrics")
async def prometheus_metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

# ─────────────────────────────────────────────────────────────────────────────
# 🚀 Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # timeout_keep_alive=600 для долгих /embed запросов (согласовано)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, timeout_keep_alive=600, log_level="info")
