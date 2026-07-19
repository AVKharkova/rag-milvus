"""Admin API key protection (fail-closed when required)."""

from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException

from src.config import settings


async def require_admin_key(
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
) -> None:
    if not settings.admin_api_key:
        if settings.admin_auth_required:
            raise HTTPException(
                status_code=503,
                detail="Admin API is locked: ADMIN_API_KEY is not configured",
            )
        return
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")
