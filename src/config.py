"""RAG Milvus — lightweight config (Milvus-only, no OpenSearch/Postgres)."""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "rag_chunks"

    # Embeddings (OpenAI-compatible TEI / Infinity)
    embedder_url: str = "http://localhost:8080"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dimension: int = 384
    embedding_model_version: str = "e5-small-v1"

    # Optional reranker (leave empty to disable)
    reranker_url: str = ""
    reranking_model: str = "BAAI/bge-reranker-base"

    rrf_k: int = 60
    retrieval_candidates: int = 40
    rerank_top_k: int = 10

    # Chunking
    chunk_max_tokens: int = 512
    chunk_overlap_sentences: int = 1
    contextual_retrieval_enabled: bool = False
    contextual_retrieval_mock: bool = True
    contextual_model: str = "gpt-4o-mini"

    # LLM
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-4o-mini"
    llm_gateway_url: str = ""
    classifier_model: str = "gpt-4o-mini"

    # Optional Langfuse (disabled when placeholder)
    langfuse_public_key: str = "pk-lf-placeholder"
    langfuse_secret_key: str = "sk-lf-placeholder"
    langfuse_host: str = "http://localhost:3000"

    # Security
    guardrails_enabled: bool = True
    guardrails_check_pii: bool = False
    require_org_context: bool = False
    prefer_org_header: bool = True
    admin_api_key: str = "dev-admin-key"
    admin_auth_required: bool = True
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        parts = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return parts or ["*"]

    @property
    def openai_compatible_base_url(self) -> Optional[str]:
        url = (self.llm_gateway_url or "").rstrip("/")
        if not url:
            return None
        suffix = "/chat/completions"
        if url.endswith(suffix):
            return url[: -len(suffix)]
        if url.endswith("/v1"):
            return url
        return None


settings = Settings()

CROSS_TENANT_ORG_IDS = frozenset({"*", "all", "__all__", "everywhere"})


def is_cross_tenant_org(org_id: Optional[str]) -> bool:
    if org_id is None:
        return False
    return org_id.strip().casefold() in CROSS_TENANT_ORG_IDS


def resolve_org_id(
    *,
    header_org_id: Optional[str],
    body_org_id: Optional[str],
) -> Optional[str]:
    if settings.prefer_org_header:
        return header_org_id or body_org_id
    return body_org_id or header_org_id


import os

if settings.langfuse_public_key and settings.langfuse_public_key != "pk-lf-placeholder":
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host
