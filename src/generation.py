"""Streaming RAG generation over Milvus retrieval."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

from src.config import settings
from src.search import search_pipeline

logger = logging.getLogger(__name__)


class CustomRetriever(BaseRetriever):
    org_id: Optional[str] = None
    acl: Optional[list[str]] = None
    search_type: str = "hybrid"
    use_reranker: bool = True
    limit: int = 5
    rrf_k: int = 60

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError("Use async retrieval")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_res = await search_pipeline(
            query=query,
            limit=self.limit,
            org_id=self.org_id,
            acl=self.acl,
            search_type=self.search_type,
            use_reranker=self.use_reranker,
            rrf_k=self.rrf_k,
        )
        ranked = list(reversed(search_res.get("results") or []))
        docs: list[Document] = []
        for res in ranked:
            text = (res.get("content") or "").strip()
            if text:
                docs.append(
                    Document(page_content=text, metadata=res.get("metadata") or {})
                )
        return docs


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)
    )


DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы — AI-ассистент корпоративной базы знаний.\n"
            "Ответьте строго по контексту ниже.\n"
            "Если в контексте нет ответа — скажите: "
            "'К сожалению, в базе данных нет информации об этом'.\n\n"
            "Контекст:\n{context}",
        ),
        ("human", "Запрос: {question}"),
    ]
)


async def generate_answer_stream(
    query: str,
    org_id: Optional[str] = None,
    acl: Optional[list[str]] = None,
    *,
    limit: int = 5,
    search_type: str = "hybrid",
    use_reranker: bool = True,
    rrf_k: int = 60,
    engine: str = "milvus",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AsyncIterator[str]:
    api_key = settings.openai_api_key
    if not api_key or api_key == "sk-placeholder":
        yield "__TRACE_ID__:\n\n"
        yield "Ошибка: задайте OPENAI_API_KEY в .env."
        return

    retriever = CustomRetriever(
        org_id=org_id,
        acl=acl,
        limit=limit,
        search_type=search_type or "hybrid",
        use_reranker=bool(use_reranker),
        rrf_k=rrf_k or 60,
    )

    llm_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": settings.openai_model or "gpt-4o-mini",
        "temperature": 0.1,
    }
    if settings.openai_compatible_base_url:
        llm_kwargs["base_url"] = settings.openai_compatible_base_url

    llm = ChatOpenAI(**llm_kwargs)
    yield "__TRACE_ID__:\n\n"

    try:
        docs = await retriever._aget_relevant_documents(query, run_manager=None)  # type: ignore[arg-type]
        context_text = format_docs(docs)
    except Exception as exc:
        logger.exception("Retrieval failed")
        yield f"Ошибка retrieval: {exc}"
        return

    if not context_text.strip():
        yield "К сожалению, в базе данных нет информации об этом"
        return

    chain = DEFAULT_PROMPT | llm | StrOutputParser()
    try:
        async for chunk in chain.astream(
            {"context": context_text, "question": query}
        ):
            yield chunk
    except Exception as exc:
        logger.exception("LLM failed")
        yield f"Ошибка LLM: {exc}"
