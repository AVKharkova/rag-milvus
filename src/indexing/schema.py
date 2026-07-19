"""Pydantic schemas for Milvus chunk documents."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.config import settings


class ChunkDocument(BaseModel):
    content: str = Field(min_length=1)
    parent_content: str = Field(min_length=1)
    embedding: list[float]
    org_id: str
    acl: list[str] = Field(default_factory=lambda: ["public"])
    document_id: str
    document_title: str = ""
    source_type: str = "document"
    doc_version: int = 1
    embedding_model_version: str = "e5-small-v1"
    chunk_index: int = 0
    contextual_prefix: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("embedding")
    @classmethod
    def validate_dimension(cls, value: list[float]) -> list[float]:
        dim = settings.embedding_dimension
        if len(value) != dim:
            raise ValueError(f"embedding must be {dim}-dimensional, got {len(value)}")
        return value
