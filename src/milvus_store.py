"""Milvus hybrid store — primary (and only) vector backend for rag-milvus."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings

logger = logging.getLogger(__name__)

try:
    from pymilvus import (
        AnnSearchRequest,
        DataType,
        Function,
        FunctionType,
        MilvusClient,
        RRFRanker,
    )

    _PYMILVUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    AnnSearchRequest = None  # type: ignore[misc, assignment]
    DataType = None  # type: ignore[misc, assignment]
    Function = None  # type: ignore[misc, assignment]
    FunctionType = None  # type: ignore[misc, assignment]
    MilvusClient = None  # type: ignore[misc, assignment]
    RRFRanker = None  # type: ignore[misc, assignment]
    _PYMILVUS_AVAILABLE = False

_client: Any = None
_client_init_error: Optional[str] = None


def collection_name() -> str:
    return settings.milvus_collection


def milvus_available() -> bool:
    return get_milvus_client() is not None


def get_milvus_client() -> Any:
    global _client, _client_init_error
    if _client is not None:
        return _client
    if _client_init_error is not None:
        return None
    if not _PYMILVUS_AVAILABLE:
        _client_init_error = "pymilvus not installed"
        return None
    try:
        _client = MilvusClient(settings.milvus_uri)
        return _client
    except Exception as exc:
        _client_init_error = str(exc)
        logger.warning("Could not connect to Milvus at %s: %s", settings.milvus_uri, exc)
        return None


def init_milvus_collection() -> bool:
    client = get_milvus_client()
    if not client:
        return False
    name = collection_name()
    if client.has_collection(name):
        return True

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("org_id", DataType.VARCHAR, max_length=64, is_partition_key=True)
    schema.add_field("document_id", DataType.VARCHAR, max_length=64)
    schema.add_field("chunk_index", DataType.INT64)
    russian_analyzer = {
        "tokenizer": "standard",
        "filter": ["lowercase", {"type": "stemmer", "language": "russian"}],
    }
    schema.add_field(
        "content",
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params=russian_analyzer,
    )
    schema.add_field(
        "dense", DataType.FLOAT_VECTOR, dim=settings.embedding_dimension
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    schema.add_function(
        Function(
            name="content_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse"],
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense", index_type="AUTOINDEX", metric_type="COSINE"
    )
    index_params.add_index(
        field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
    )

    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Milvus collection '%s' created (dim=%s)", name, settings.embedding_dimension)
    return True


def _acl_allows(doc_acl: List[str], required: Optional[List[str]]) -> bool:
    if not required:
        return True
    return bool(set(doc_acl or []).intersection(required))


async def bulk_insert_milvus(docs: list[tuple[str, Any]]) -> dict[str, Any]:
    client = get_milvus_client()
    if not client:
        return {"indexed": 0, "errors": 0, "skipped": True}
    if not docs:
        return {"indexed": 0, "errors": 0, "skipped": False}

    def _insert() -> dict[str, Any]:
        init_milvus_collection()
        data: list[dict[str, Any]] = []
        for chunk_id, chunk_doc in docs:
            acl = list(getattr(chunk_doc, "acl", None) or ["public"])
            data.append(
                {
                    "pk": str(chunk_id),
                    "org_id": str(chunk_doc.org_id),
                    "document_id": str(chunk_doc.document_id),
                    "chunk_index": int(chunk_doc.chunk_index),
                    "content": chunk_doc.content,
                    "parent_content": chunk_doc.parent_content,
                    "acl_json": json.dumps(acl, ensure_ascii=False),
                    "document_title": getattr(chunk_doc, "document_title", "") or "",
                    "dense": chunk_doc.embedding,
                }
            )
        res = client.upsert(collection_name=collection_name(), data=data)
        upserted = int(
            res.get("upsert_count", len(data)) if isinstance(res, dict) else len(data)
        )
        return {"indexed": upserted, "errors": 0, "skipped": False}

    return await asyncio.to_thread(_insert)


async def search_milvus_hybrid(
    query: str,
    embedding: List[float],
    limit: int,
    org_id: Optional[str],
    acl: Optional[List[str]],
    rrf_k: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: list[str] = []
    client = get_milvus_client()
    if not client:
        return [], [f"Milvus unavailable ({_client_init_error or 'no client'})"]
    if not embedding:
        return [], ["Milvus hybrid requires a query embedding."]

    def _search() -> list[dict[str, Any]]:
        name = collection_name()
        if not client.has_collection(name):
            init_milvus_collection()
            return []

        expr = ""
        if org_id:
            safe = str(org_id).replace('"', "").replace("'", "").replace("\\", "")
            expr = f'org_id == "{safe}"'

        fetch_n = max(limit * 4, limit)
        dense_kwargs: dict[str, Any] = {
            "data": [embedding],
            "anns_field": "dense",
            "param": {"metric_type": "COSINE"},
            "limit": fetch_n,
        }
        sparse_kwargs: dict[str, Any] = {
            "data": [query],
            "anns_field": "sparse",
            "param": {"metric_type": "BM25"},
            "limit": fetch_n,
        }
        if expr:
            dense_kwargs["expr"] = expr
            sparse_kwargs["expr"] = expr

        res = client.hybrid_search(
            collection_name=name,
            reqs=[AnnSearchRequest(**dense_kwargs), AnnSearchRequest(**sparse_kwargs)],
            ranker=RRFRanker(rrf_k if rrf_k is not None else settings.rrf_k),
            limit=fetch_n,
            output_fields=[
                "pk",
                "content",
                "parent_content",
                "document_id",
                "chunk_index",
                "org_id",
                "acl_json",
                "document_title",
            ],
        )

        results: list[dict[str, Any]] = []
        for hit in res[0]:
            entity = hit.get("entity") or {}
            try:
                doc_acl = json.loads(entity.get("acl_json") or "[]")
                if not isinstance(doc_acl, list):
                    doc_acl = ["public"]
            except json.JSONDecodeError:
                doc_acl = ["public"]
            if not _acl_allows(doc_acl, acl):
                continue

            parent = entity.get("parent_content") or ""
            indexed = entity.get("content") or ""
            results.append(
                {
                    "id": hit.get("id") or entity.get("pk"),
                    "content": parent or indexed,
                    "indexed_content": indexed,
                    "score": hit.get("distance", 0.0),
                    "metadata": {
                        "document_id": entity.get("document_id"),
                        "chunk_index": entity.get("chunk_index"),
                        "org_id": entity.get("org_id"),
                        "acl": doc_acl,
                        "document_title": entity.get("document_title") or "",
                        "engine": "milvus",
                    },
                }
            )
            if len(results) >= limit * 2:
                break
        return results

    try:
        return await asyncio.to_thread(_search), warnings
    except Exception as exc:
        logger.error("Milvus hybrid search failed: %s", exc)
        return [], [f"Milvus hybrid failed: {exc}"]
