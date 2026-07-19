"""FastAPI entrypoint — Milvus-only RAG service."""

from __future__ import annotations

import asyncio
import platform
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import resolve_org_id, settings
from src.generation import generate_answer_stream
from src.indexing.ingest_tasks import create_task, get_task
from src.indexing.ingestion import (
    ingest_parsed_document,
    ingest_upload,
    run_ingest_task,
    run_ingest_upload_task,
)
from src.indexing.parser import _parse_plain_text
from src.milvus_store import init_milvus_collection, milvus_available
from src.search import search_pipeline
from src.security.admin_auth import require_admin_key
from src.security.middleware import enforce_query_guardrails

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if milvus_available():
            init_milvus_collection()
            app.state.milvus_bootstrap = {"status": "ok"}
        else:
            app.state.milvus_bootstrap = {
                "status": "unavailable",
                "detail": "Cannot reach Milvus — check MILVUS_URI",
            }
    except Exception as exc:
        app.state.milvus_bootstrap = {"status": "error", "detail": str(exc)}
    yield


app = FastAPI(
    title="RAG Milvus",
    description="Lightweight RAG service: hybrid search on Milvus only",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=50)
    org_id: Optional[str] = None
    acl: Optional[List[str]] = None
    search_type: Optional[str] = "hybrid"
    use_reranker: Optional[bool] = True
    rrf_k: Optional[int] = 60
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class SearchResult(BaseModel):
    id: Optional[str] = None
    content: str
    relevance_score: Optional[float] = None
    score: Optional[float] = None
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: List[SearchResult]
    warnings: List[str]
    metrics: Optional[dict] = None


class IngestTextRequest(BaseModel):
    title: str
    text: str
    org_id: str
    acl: Optional[List[str]] = None
    doc_version: int = 1
    sync: bool = True


@app.post("/v1/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    x_org_id: Optional[str] = Header(default=None, alias="X-Org-Id"),
):
    enforce_query_guardrails(req.query)
    org_id = resolve_org_id(header_org_id=x_org_id, body_org_id=req.org_id)
    try:
        return await search_pipeline(
            query=req.query,
            limit=req.limit,
            org_id=org_id,
            acl=req.acl,
            search_type=req.search_type or "hybrid",
            use_reranker=bool(req.use_reranker),
            rrf_k=req.rrf_k or 60,
            session_id=req.session_id,
            user_id=req.user_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/ask")
async def ask(
    req: SearchRequest,
    x_org_id: Optional[str] = Header(default=None, alias="X-Org-Id"),
):
    enforce_query_guardrails(req.query)
    org_id = resolve_org_id(header_org_id=x_org_id, body_org_id=req.org_id)
    return StreamingResponse(
        generate_answer_stream(
            query=req.query,
            org_id=org_id,
            acl=req.acl,
            limit=req.limit,
            search_type=req.search_type or "hybrid",
            use_reranker=bool(req.use_reranker),
            rrf_k=req.rrf_k or 60,
            session_id=req.session_id,
            user_id=req.user_id,
        ),
        media_type="text/plain; charset=utf-8",
    )


@app.post("/v1/admin/ingest/text")
async def admin_ingest_text(
    req: IngestTextRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_admin_key),
):
    parsed = _parse_plain_text(req.text, title=req.title)
    try:
        if req.sync:
            result = await ingest_parsed_document(
                parsed,
                org_id=req.org_id,
                acl=req.acl,
                doc_version=req.doc_version,
            )
            if result.get("status") not in ("ok", "empty"):
                raise HTTPException(status_code=500, detail=result)
            return result
        task_id = str(uuid.uuid4())
        create_task(task_id, title=req.title, org_id=req.org_id)
        background_tasks.add_task(
            run_ingest_task,
            task_id,
            parsed,
            org_id=req.org_id,
            acl=req.acl,
            doc_version=req.doc_version,
        )
        return {"status": "accepted", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/admin/ingest/file")
async def admin_ingest_file(
    background_tasks: BackgroundTasks,
    org_id: str = Form(...),
    file: UploadFile = File(...),
    acl: Optional[str] = Form(default=None),
    sync: bool = Form(default=True),
    _: None = Depends(require_admin_key),
):
    content = await file.read()
    acl_list = [a.strip() for a in acl.split(",")] if acl else None
    filename = file.filename or "upload.txt"
    try:
        if sync:
            result = await ingest_upload(
                filename=filename,
                content=content,
                org_id=org_id,
                acl=acl_list,
                content_type=file.content_type,
            )
            if result.get("status") not in ("ok", "empty"):
                raise HTTPException(status_code=500, detail=result)
            return result
        task_id = str(uuid.uuid4())
        create_task(task_id, title=filename, org_id=org_id)
        background_tasks.add_task(
            run_ingest_upload_task,
            task_id,
            filename=filename,
            content=content,
            org_id=org_id,
            acl=acl_list,
            content_type=file.content_type,
        )
        return {"status": "accepted", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/admin/ingest/tasks/{task_id}")
async def admin_ingest_task_status(task_id: str, _: None = Depends(require_admin_key)):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    return task


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "rag-milvus",
        "version": "0.1.0",
        "features": {
            "milvus": milvus_available(),
            "reranker": bool(settings.reranker_url),
            "guardrails": settings.guardrails_enabled,
            "admin_auth": bool(settings.admin_api_key),
        },
        "milvus_bootstrap": getattr(app.state, "milvus_bootstrap", None),
        "milvus_uri": settings.milvus_uri,
        "embedding_dimension": settings.embedding_dimension,
    }


app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
