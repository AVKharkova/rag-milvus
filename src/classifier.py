import json
import logging

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


async def classify_query(query: str) -> bool:
    """Adaptive RAG: determine if query needs knowledge-base retrieval."""
    prompt = (
        "You are a strict routing assistant. Should the user's input be routed to a RAG search engine (YES) or handled as generic chitchat (NO)?\n"
        "Rules:\n"
        "1. If the input is a simple greeting ('Привет', 'Hello', 'Здрасьте'), a farewell, a 'thank you', or a completely empty/meaningless string -> Output NO.\n"
        "2. If the input is ANY question (asking for facts, definitions, history, names, people, movies, science, errors, corporate data, or literally ANY query that requires information) -> Output YES.\n"
        "3. IF IN DOUBT, ALWAYS OUTPUT YES.\n"
        "Reply ONLY with a single word: 'YES' or 'NO'.\n\n"
        f"User query: {query}"
    )

    request = {
        "model": settings.classifier_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 5,
        "stream": True,
    }

    headers = {"Content-Type": "application/json"}
    if settings.openai_api_key and settings.openai_api_key != "sk-placeholder":
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"

    full = ""
    try:
        url = settings.openai_compatible_base_url
        if not url:
            url = "https://api.openai.com/v1"
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            async with client.stream(
                "POST", f"{url}/chat/completions", json=request, headers=headers
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[len("data: ") :].strip()
                    if not chunk or chunk == "[DONE]":
                        continue
                    try:
                        data = json.loads(chunk)
                        delta = data["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            full += delta
                    except Exception:
                        continue
    except Exception as exc:
        logger.warning("Query classification failed: %s", exc)
        return True

    answer = full.strip().upper()
    logger.warning("Classifier raw LLM answer: %r for query: %r", answer, query)
    return bool(answer.startswith("YES"))
