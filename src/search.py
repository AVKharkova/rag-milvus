"""Milvus-only hybrid retrieval: embed → hybrid search → optional rerank."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from src.config import is_cross_tenant_org, settings
from src.milvus_store import search_milvus_hybrid

try:
    from langfuse.decorators import observe
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


async def _get_embedding(query: str) -> tuple[list[float], list[str]]:
    warnings: list[str] = []
    text = query

    if settings.is_yandex_embeddings:
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        headers = {"Authorization": f"Api-Key {settings.yandex_api_key}"}
        model_uri = f"emb://{settings.yandex_folder_id}/text-search-query/latest"
        async with httpx.AsyncClient() as client:
            try:
                res = await client.post(
                    url,
                    headers=headers,
                    json={"modelUri": model_uri, "text": text},
                    timeout=60.0
                )
                res.raise_for_status()
                return res.json()["embedding"], warnings
            except Exception as exc:
                logger.warning("Yandex Query Embedding failed: %s", exc)
                return [], ["Vector search skipped - Yandex embedding unavailable"]

    # TEI/OpenAI fallback
    if "e5" in settings.embedding_model.lower() and not query.startswith("query:"):
        text = f"query: {query}"
    async with httpx.AsyncClient() as client:
        try:
            base = settings.embedder_url.rstrip("/")
            last_exc: Exception | None = None
            for path in ("/v1/embeddings", "/embeddings"):
                try:
                    response = await client.post(
                        f"{base}{path}",
                        json={"model": settings.embedding_model, "input": [text]},
                        timeout=60.0,
                    )
                    response.raise_for_status()
                    return response.json()["data"][0]["embedding"], warnings
                except Exception as exc:
                    last_exc = exc
            raise last_exc or RuntimeError("embedding failed")
        except Exception as exc:
            logger.warning("Embedding request failed: %s", exc)
            return [], ["Vector search skipped - embedding unavailable"]


async def _rerank(
    query: str,
    candidates: list[tuple[float, str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not settings.reranker_url:
        return [
            {
                "id": doc["id"],
                "content": doc["content"],
                "score": score,
                "metadata": doc.get("metadata", {}),
            }
            for score, _, doc in candidates[: settings.rerank_top_k]
        ], ["Reranker disabled (RERANKER_URL empty)"]

    docs = [doc.get("indexed_content") or doc["content"] for _, _, doc in candidates]
    async with httpx.AsyncClient() as client:
        try:
            req_json = {
                "model": settings.reranking_model,
                "query": query,
                "documents": docs,
            }
            logger.warning("Reranker payload: %s", req_json)
            response = await client.post(
                f"{settings.reranker_url.rstrip('/')}/rerank",
                json=req_json,
                timeout=60.0,
            )
            response.raise_for_status()
            rerank_data = response.json()
            results = []
            for item in rerank_data.get("results") or []:
                idx = item["index"]
                _, _, doc = candidates[idx]
                results.append(
                    {
                        "id": doc["id"],
                        "content": doc["content"],
                        "relevance_score": item.get("relevance_score") if item.get("relevance_score") is not None else item.get("score"),
                        "metadata": doc.get("metadata", {}),
                    }
                )
            return results, warnings
        except Exception as exc:
            err_body = ""
            if isinstance(exc, httpx.HTTPStatusError):
                err_body = f", Body: {exc.response.text}"
            logger.warning("Reranking failed: %s%s", exc, err_body)
            warnings.append("Reranking failed - using search order")
            return [
                {
                    "id": doc["id"],
                    "content": doc["content"],
                    "score": rrf_score,
                    "metadata": doc.get("metadata", {}),
                }
                for rrf_score, _, doc in candidates[: settings.rerank_top_k]
            ], warnings


@observe(name="search_pipeline")
async def search_pipeline(
    query: str,
    limit: int = 10,
    org_id: Optional[str] = None,
    acl: Optional[list[str]] = None,
    search_type: str = "hybrid",
    use_reranker: bool = True,
    rrf_k: int = 60,
    engine: str = "milvus",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    metrics: dict[str, float] = {}
    t_start = time.perf_counter()

    if session_id or user_id:
        try:
            from langfuse.decorators import langfuse_context

            langfuse_context.update_current_trace(session_id=session_id, user_id=user_id)
        except Exception:
            pass

    if is_cross_tenant_org(org_id):
        org_id = None
    elif settings.require_org_context and not org_id:
        return {
            "results": [],
            "warnings": ["Missing org_id — fail-closed policy"],
            "metrics": {"total_time_ms": 0.0},
        }

    if search_type != "hybrid":
        warnings.append(
            f"Milvus service only implements hybrid; ignoring search_type={search_type!r}."
        )

    t_embed = time.perf_counter()
    embedding, embed_warnings = await _get_embedding(query)
    warnings.extend(embed_warnings)
    metrics["embedding_time_ms"] = round((time.perf_counter() - t_embed) * 1000, 2)

    t_ret = time.perf_counter()
    milvus_results, ret_warn = await search_milvus_hybrid(
        query, embedding, limit, org_id, acl, rrf_k=rrf_k or settings.rrf_k
    )
    warnings.extend(ret_warn)
    candidates = [(d.get("score", 0.0), d["id"], d) for d in milvus_results]
    metrics["retrieval_time_ms"] = round((time.perf_counter() - t_ret) * 1000, 2)
    metrics["engine"] = "milvus"

    seen: set[str] = set()
    unique = []
    for cand in candidates:
        c = cand[2]["content"]
        if c not in seen:
            seen.add(c)
            unique.append(cand)
    candidates = unique
    metrics["candidate_count"] = len(candidates)

    if not candidates:
        metrics["total_time_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return {"results": [], "warnings": warnings, "metrics": metrics}

    t_rr = time.perf_counter()
    if use_reranker and settings.reranker_url:
        final, rr_warn = await _rerank(query, candidates)
        warnings.extend(rr_warn)
        top = final[: min(limit, settings.rerank_top_k)]
    else:
        if use_reranker and not settings.reranker_url:
            warnings.append("Reranker requested but RERANKER_URL is empty")
        top = [
            {
                "id": doc["id"],
                "content": doc["content"],
                "score": score,
                "metadata": doc.get("metadata", {}),
            }
            for score, _, doc in candidates[:limit]
        ]
    metrics["rerank_time_ms"] = round((time.perf_counter() - t_rr) * 1000, 2)
    metrics["total_time_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

    # Reverse for LLM lost-in-the-middle mitigation
    return {"results": top[::-1], "warnings": warnings, "metrics": metrics}
