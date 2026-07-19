"""Milvus-only ingestion: parse → chunk → embed → Milvus."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from src.config import settings
from src.indexing.chunking import chunk_text_sentence_window
from src.indexing.contextual import build_contextual_content, generate_chunk_context
from src.indexing.parser import ParsedDocument, parse_upload
from src.indexing.schema import ChunkDocument
from src.milvus_store import bulk_insert_milvus

logger = logging.getLogger(__name__)


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    model = settings.embedding_model.lower()
    if "e5" in model:
        texts = [t if t.startswith("passage:") else f"passage: {t}" for t in texts]

    base = settings.embedder_url.rstrip("/")
    # TEI OpenAI-compat: /v1/embeddings; Infinity: /embeddings
    urls = [f"{base}/v1/embeddings", f"{base}/embeddings"]
    last_exc: Exception | None = None
    async with httpx.AsyncClient() as client:
        for url in urls:
            try:
                response = await client.post(
                    url,
                    json={"model": settings.embedding_model, "input": texts},
                    timeout=120.0,
                )
                response.raise_for_status()
                return [item["embedding"] for item in response.json()["data"]]
            except Exception as exc:
                last_exc = exc
                continue
    raise RuntimeError(f"Embedding failed: {last_exc}")


async def ingest_parsed_document(
    parsed: ParsedDocument,
    *,
    org_id: str,
    acl: Optional[list[str]] = None,
    doc_version: int = 1,
    document_id: Optional[str] = None,
) -> dict[str, Any]:
    acl = acl or [f"org:{org_id}"]
    document_id = document_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    excerpt = parsed.text[:1200]

    window_chunks = chunk_text_sentence_window(
        parsed.text,
        max_tokens=settings.chunk_max_tokens,
        overlap_sentences=settings.chunk_overlap_sentences,
    )
    if not window_chunks:
        return {"document_id": document_id, "chunks_created": 0, "status": "empty"}

    semaphore = asyncio.Semaphore(8)

    async def process_item(item):
        async with semaphore:
            context = await generate_chunk_context(
                title=parsed.title,
                chunk=item.sentence,
                document_excerpt=excerpt,
            )
            indexed_text = build_contextual_content(context, item.sentence)
            return (indexed_text, item.parent_window, context, item.chunk_index)

    contextualized = await asyncio.gather(*[process_item(i) for i in window_chunks])
    embeddings = await _embed_texts([t for t, _, _, _ in contextualized])

    milvus_docs: list[tuple[str, ChunkDocument]] = []
    for (indexed_text, parent_window, context, chunk_index), embedding in zip(
        contextualized, embeddings
    ):
        chunk_id = str(uuid.uuid4())
        milvus_docs.append(
            (
                chunk_id,
                ChunkDocument(
                    content=indexed_text,
                    parent_content=parent_window,
                    embedding=embedding,
                    org_id=org_id,
                    acl=acl,
                    document_id=document_id,
                    document_title=parsed.title,
                    source_type=parsed.source_type,
                    doc_version=doc_version,
                    embedding_model_version=settings.embedding_model_version,
                    chunk_index=chunk_index,
                    contextual_prefix=context,
                    created_at=now,
                    updated_at=now,
                ),
            )
        )

    milvus_stats = await bulk_insert_milvus(milvus_docs)
    ok = milvus_stats.get("errors", 0) == 0 and not milvus_stats.get("skipped")
    return {
        "document_id": document_id,
        "title": parsed.title,
        "chunks_created": len(milvus_docs),
        "indexed": milvus_stats.get("indexed", 0),
        "errors": milvus_stats.get("errors", 0),
        "milvus": milvus_stats,
        "parser": parsed.metadata.get("parser"),
        "status": "ok" if ok else ("skipped" if milvus_stats.get("skipped") else "error"),
    }


async def ingest_upload(
    *,
    filename: str,
    content: bytes,
    org_id: str,
    acl: Optional[list[str]] = None,
    content_type: Optional[str] = None,
) -> dict[str, Any]:
    parsed = parse_upload(filename=filename, content=content, content_type=content_type)
    return await ingest_parsed_document(parsed, org_id=org_id, acl=acl)


async def _run_tracked_task(task_id: str, work) -> None:
    from src.indexing.ingest_tasks import set_task_done, set_task_error, set_task_running

    set_task_running(task_id)
    try:
        set_task_done(task_id, await work)
    except Exception as exc:
        logger.exception("Background ingest task %s failed", task_id)
        set_task_error(task_id, str(exc))


async def run_ingest_task(task_id: str, parsed: ParsedDocument, *, org_id: str, acl=None, doc_version: int = 1) -> None:
    await _run_tracked_task(
        task_id,
        ingest_parsed_document(parsed, org_id=org_id, acl=acl, doc_version=doc_version),
    )


async def run_ingest_upload_task(
    task_id: str,
    *,
    filename: str,
    content: bytes,
    org_id: str,
    acl=None,
    content_type: Optional[str] = None,
) -> None:
    await _run_tracked_task(
        task_id,
        ingest_upload(
            filename=filename,
            content=content,
            org_id=org_id,
            acl=acl,
            content_type=content_type,
        ),
    )
