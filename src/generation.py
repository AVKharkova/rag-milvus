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

try:
    from langfuse.decorators import observe
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

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


def _prompt_has_context(prompt: ChatPromptTemplate) -> bool:
    try:
        return "context" in set(prompt.input_variables)
    except Exception:
        return False


def get_rag_prompt() -> tuple[ChatPromptTemplate, Any]:
    """Fetch prompt from Langfuse or fallback to default."""
    try:
        from langfuse import Langfuse

        langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        lf_prompt = langfuse_client.get_prompt("rag-system-prompt")
        compiled: Any = lf_prompt.get_langchain_prompt()

        if isinstance(compiled, str):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", compiled),
                    ("human", "{question}"),
                ]
            )
        elif isinstance(compiled, list):
            prompt = ChatPromptTemplate.from_messages(compiled)
        else:
            logger.warning(
                "Unexpected Langfuse prompt type %s — using default", type(compiled)
            )
            return DEFAULT_PROMPT, None

        if not _prompt_has_context(prompt):
            logger.warning("Langfuse prompt 'rag-system-prompt' has no {context}")
            return DEFAULT_PROMPT, None
        return prompt, lf_prompt
    except Exception as exc:
        logger.warning("Failed to fetch prompt from Langfuse: %s", exc)
        return DEFAULT_PROMPT, None



@observe(name="generate_answer_stream", as_type="generation")
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

    trace_id = "unknown"
    langfuse_handler = None
    try:
        from langfuse.decorators import langfuse_context
        trace_id = langfuse_context.get_current_trace_id() or "unknown"
        langfuse_handler = langfuse_context.get_current_langchain_handler()
        if session_id or user_id:
            langfuse_context.update_current_trace(session_id=session_id, user_id=user_id)
    except Exception:
        pass
        
    yield f"__TRACE_ID__:{trace_id}\n\n"

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

    prompt_template, lf_prompt = get_rag_prompt()
    if lf_prompt:
        try:
            from langfuse.decorators import langfuse_context
            langfuse_context.update_current_observation(prompt=lf_prompt)
        except Exception:
            pass

    chain = prompt_template | llm | StrOutputParser()
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    try:
        async for chunk in chain.astream(
            {"context": context_text, "question": query},
            config=config
        ):
            yield chunk
    except Exception as exc:
        logger.exception("LLM failed")
        yield f"Ошибка LLM: {exc}"
    finally:
        try:
            from langfuse.decorators import langfuse_context
            langfuse_context.flush()
        except Exception:
            pass


@observe(name="generate_chat_stream", as_type="generation")
async def generate_chat_stream(
    query: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AsyncIterator[str]:
    """Simple chat endpoint without RAG."""
    api_key = settings.openai_api_key
    if not api_key or api_key == "sk-placeholder":
        yield "Ошибка: задайте OPENAI_API_KEY в .env."
        return

    llm_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": settings.openai_model or "gpt-4o-mini",
        "temperature": 0.7,
    }
    if settings.openai_compatible_base_url:
        llm_kwargs["base_url"] = settings.openai_compatible_base_url

    llm = ChatOpenAI(**llm_kwargs)
    
    trace_id = "unknown"
    langfuse_handler = None
    try:
        from langfuse.decorators import langfuse_context
        trace_id = langfuse_context.get_current_trace_id() or "unknown"
        langfuse_handler = langfuse_context.get_current_langchain_handler()
        if session_id or user_id:
            langfuse_context.update_current_trace(session_id=session_id, user_id=user_id)
    except Exception:
        pass
        
    yield f"__TRACE_ID__:{trace_id}\n\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Вы — AI-ассистент корпоративной базы знаний. Ответьте на общий вопрос пользователя или поприветствуйте его."),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    try:
        async for chunk in chain.astream({"question": query}, config=config):
            yield chunk
    except Exception as exc:
        logger.exception("LLM failed")
        yield f"Ошибка LLM: {exc}"
    finally:
        try:
            from langfuse.decorators import langfuse_context
            langfuse_context.flush()
        except Exception:
            pass
