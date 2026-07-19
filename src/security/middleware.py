"""Query guardrails as a FastAPI-safe helper (no BaseHTTPMiddleware).

BaseHTTPMiddleware breaks StreamingResponse (/v1/ask) with:
RuntimeError: Unexpected message received: http.request
"""

from __future__ import annotations

from fastapi import HTTPException

from src.config import settings
from src.security.guardrails import check_content


def enforce_query_guardrails(query: str) -> None:
    """Raise HTTP 400 if query fails injection/PII checks."""
    if not settings.guardrails_enabled:
        return
    if not isinstance(query, str):
        raise HTTPException(status_code=400, detail="query must be a string")

    result = check_content(query, check_pii=settings.guardrails_check_pii)
    if not result.allowed:
        raise HTTPException(
            status_code=400,
            detail={
                "message": result.reason,
                "violations": result.violations,
            },
        )
