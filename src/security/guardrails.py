"""Guardrails: prompt injection and PII detection (air-gap, no external API)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?prior",
        r"system\s+prompt",
        r"jailbreak",
        r"<\s*script",
        r"```\s*system",
    ]
]

PII_PATTERNS = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone_ru", re.compile(r"\b(?:\+7|8)[\s(-]?\d{3}[\s)-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b")),
    ("passport_ru", re.compile(r"\b\d{4}\s?\d{6}\b")),
]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: Optional[str] = None
    violations: Optional[list[str]] = None


def check_content(text: str, *, check_pii: bool = True) -> GuardrailResult:
    violations: list[str] = []

    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            violations.append("prompt_injection")

    if check_pii:
        for name, pattern in PII_PATTERNS:
            if pattern.search(text):
                violations.append(f"pii:{name}")

    if violations:
        return GuardrailResult(
            allowed=False,
            reason="Content blocked by guardrails",
            violations=violations,
        )
    return GuardrailResult(allowed=True)
