"""FastAPI routes and lazy Agent initialization."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config.settings import MILVUS_CONFIG, PG_CONFIG
from src.agent.graph_definition import NutritionAgent
from src.indexing.llm_client import get_default_client
from src.memory.memory_manager import MemoryManager
from src.observability.runtime_logger import RuntimeLogger
from src.storage.milvus_client import MilvusClient
from src.storage.pg_client import PostgreSQLClient
from src.user_content.classifier import UserContentClassifier
from src.user_content.models import UserContentType
from src.user_content.service import UserContentService

logger = logging.getLogger(__name__)

router = APIRouter()
_agent_lock = asyncio.Lock()
_agent_run_lock = asyncio.Lock()
_content_lock = asyncio.Lock()
_agent: NutritionAgent | None = None
_content_service: UserContentService | None = None
_runtime_loggers: dict[int, RuntimeLogger] = {}


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"
    user_id: str = "default"
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    request_id: str = ""
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    intent: str = ""
    intent_confidence: float = 0.0
    intent_reason: str = ""
    session_id: str = "default"
    user_id: str = "default"
    planned_routes: list[str] = Field(default_factory=list)
    executed_routes: list[str] = Field(default_factory=list)
    fallback_routes: list[str] = Field(default_factory=list)
    route_errors: list[dict[str, Any]] = Field(default_factory=list)
    route_status: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)
    route_context: dict[str, Any] = Field(default_factory=dict)
    route_decision: dict[str, Any] = Field(default_factory=dict)
    personalization_policy: dict[str, Any] = Field(default_factory=dict)
    semantic_discovered_entities: list[dict[str, Any]] = Field(default_factory=list)
    safety_warnings: list[dict[str, Any]] = Field(default_factory=list)
    route_reason: dict[str, Any] = Field(default_factory=dict)
    has_profile_signal: bool = False
    evidence_count: int = 0
    retrieved_context_ids: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    reranked_evidence: list[dict[str, Any]] = Field(default_factory=list)
    saved_content: dict[str, Any] = Field(default_factory=dict)
    runtime_logged: bool = False
    latency_ms: int = 0


class SaveUserContentRequest(BaseModel):
    content: str = Field(..., min_length=1)
    content_type: UserContentType | None = None
    title: str = ""
    session_id: str = "default"
    user_id: str = "default"
    visibility: str = "private"


class SaveUserContentResponse(BaseModel):
    saved: bool
    content_type: str = ""
    title: str = ""
    source_doc_id: str = ""
    storage: dict[str, Any] = Field(default_factory=dict)
    classification: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", agent_ready=_agent is not None)


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    request_id = RuntimeLogger.new_request_id()
    start_ms = RuntimeLogger.now_ms()

    maybe_saved = await _try_save_explicit_user_content(request)
    if maybe_saved is not None:
        answer = (
            f"已保存为 {maybe_saved.content_type}: {maybe_saved.title}"
            if maybe_saved.saved
            else "我还不能确定这段内容属于哪类可保存数据，请说明是菜谱、饮食计划、训练计划、饮食记录、身体指标还是体检报告。"
        )
        latency_ms = RuntimeLogger.now_ms() - start_ms
        result = {
            "answer": answer,
            "citations": [],
            "session_id": request.session_id,
            "user_id": request.user_id,
            "intent": "content_save",
            "intent_confidence": 1.0 if maybe_saved.saved else 0.4,
            "planned_routes": [],
            "executed_routes": [],
            "fallback_routes": [],
            "route_status": {},
            "route_errors": [],
            "route_decision": {},
            "trace": {},
            "reranked_evidence": [],
            "personalization_policy": {},
        }
        runtime_logged = await _log_runtime_best_effort(
            pg=_pg_from_content_service(),
            request_id=request_id,
            query=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
            result=result,
            latency_ms=latency_ms,
            status="success" if maybe_saved.saved else "unsupported_content",
        )
        return ChatResponse(
            request_id=request_id,
            answer=answer,
            session_id=request.session_id,
            user_id=request.user_id,
            intent="content_save",
            intent_confidence=1.0 if maybe_saved.saved else 0.4,
            intent_reason="用户明确要求保存个人内容",
            saved_content=maybe_saved.to_dict(),
            runtime_logged=runtime_logged,
            latency_ms=latency_ms,
        )

    agent = await get_agent()
    history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    try:
        async with _agent_run_lock:
            result = await asyncio.to_thread(
                agent.run,
                request.message,
                history,
                request.session_id,
                request.user_id,
            )
    except Exception as exc:
        logger.exception("Agent chat failed")
        latency_ms = RuntimeLogger.now_ms() - start_ms
        await _log_runtime_best_effort(
            pg=getattr(agent, "pg", None),
            request_id=request_id,
            query=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
            result={
                "answer": "",
                "session_id": request.session_id,
                "user_id": request.user_id,
                "reranked_evidence": [],
                "personalization_policy": {},
            },
            latency_ms=latency_ms,
            status="error",
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Agent chat failed: {exc}") from exc

    latency_ms = RuntimeLogger.now_ms() - start_ms
    runtime_logged = await _log_runtime_best_effort(
        pg=getattr(agent, "pg", None),
        request_id=request_id,
        query=request.message,
        session_id=request.session_id,
        user_id=request.user_id,
        result=result,
        latency_ms=latency_ms,
        status="success",
    )

    return ChatResponse(
        request_id=request_id,
        answer=result.get("answer", ""),
        citations=result.get("citations", []) or [],
        entities=result.get("entities", []) or [],
        intent=result.get("intent", "") or "",
        intent_confidence=float(result.get("intent_confidence", 0.0) or 0.0),
        intent_reason=result.get("intent_reason", "") or "",
        session_id=result.get("session_id", request.session_id),
        user_id=result.get("user_id", request.user_id),
        planned_routes=result.get("planned_routes", []) or [],
        executed_routes=result.get("executed_routes", []) or [],
        fallback_routes=result.get("fallback_routes", []) or [],
        route_errors=result.get("route_errors", []) or [],
        route_status=result.get("route_status", {}) or {},
        trace=result.get("trace", {}) or {},
        route_context=result.get("route_context", {}) or {},
        route_decision=result.get("route_decision", {}) or {},
        personalization_policy=result.get("personalization_policy", {}) or {},
        semantic_discovered_entities=result.get("semantic_discovered_entities", []) or [],
        safety_warnings=result.get("safety_warnings", []) or [],
        route_reason=result.get("route_reason", {}) or {},
        has_profile_signal=bool(result.get("has_profile_signal", False)),
        evidence_count=int(result.get("evidence_count", 0) or 0),
        retrieved_context_ids=result.get("retrieved_context_ids", []) or [],
        contexts=result.get("contexts", []) or [],
        reranked_evidence=result.get("reranked_evidence", []) or [],
        runtime_logged=runtime_logged,
        latency_ms=latency_ms,
    )


@router.post("/api/user-content/save", response_model=SaveUserContentResponse)
async def save_user_content(request: SaveUserContentRequest) -> SaveUserContentResponse:
    service = await get_content_service()
    try:
        result = await asyncio.to_thread(
            service.save_text,
            user_id=request.user_id,
            session_id=request.session_id,
            content=request.content,
            content_type=request.content_type.value if request.content_type else None,
            title=request.title,
            visibility=request.visibility,
        )
    except Exception as exc:
        logger.exception("Save user content failed")
        raise HTTPException(status_code=500, detail=f"Save user content failed: {exc}") from exc
    return SaveUserContentResponse(**result.to_dict())


async def get_agent() -> NutritionAgent:
    global _agent
    if _agent is not None:
        return _agent

    async with _agent_lock:
        if _agent is not None:
            return _agent
        try:
            _agent = await asyncio.to_thread(build_agent)
        except Exception as exc:
            logger.exception("Agent initialization failed")
            raise HTTPException(
                status_code=503,
                detail=f"Agent initialization failed: {exc}",
            ) from exc
        return _agent


async def get_content_service() -> UserContentService:
    global _content_service
    if _content_service is not None:
        return _content_service

    async with _content_lock:
        if _content_service is not None:
            return _content_service
        try:
            _content_service = await asyncio.to_thread(UserContentService)
        except Exception as exc:
            logger.exception("User content service initialization failed")
            raise HTTPException(
                status_code=503,
                detail=f"User content service initialization failed: {exc}",
            ) from exc
        return _content_service


async def _try_save_explicit_user_content(request: ChatRequest):
    classifier = UserContentClassifier()
    if not classifier.is_explicit_save_request(request.message):
        return None
    classification = classifier.classify(request.message)
    if classification.content_type is None:
        return None
    service = await get_content_service()
    return await asyncio.to_thread(
        service.save_text,
        user_id=request.user_id,
        session_id=request.session_id,
        content=request.message,
        content_type=classification.content_type.value,
        title="",
        visibility="private",
    )


async def _log_runtime_best_effort(
    *,
    pg: PostgreSQLClient | None,
    request_id: str,
    query: str,
    session_id: str,
    user_id: str,
    result: dict[str, Any],
    latency_ms: int,
    status: str,
    error: str = "",
) -> bool:
    if pg is None:
        return False
    logger_instance = _runtime_logger_for(pg)
    return await asyncio.to_thread(
        logger_instance.log_chat,
        request_id=request_id,
        query=query,
        session_id=session_id,
        user_id=user_id,
        result=result,
        latency_ms=latency_ms,
        status=status,
        error=error,
    )


def _runtime_logger_for(pg: PostgreSQLClient) -> RuntimeLogger:
    key = id(pg)
    if key not in _runtime_loggers:
        _runtime_loggers[key] = RuntimeLogger(pg)
    return _runtime_loggers[key]


def _pg_from_content_service() -> PostgreSQLClient | None:
    storage = getattr(_content_service, "storage", None)
    return getattr(storage, "pg_client", None)


def build_agent() -> NutritionAgent:
    pg = _construct(PostgreSQLClient, PG_CONFIG)
    milvus = _construct(MilvusClient, MILVUS_CONFIG)
    llm = get_default_client()
    memory = MemoryManager(pg=pg, milvus=milvus)
    return NutritionAgent(
        pg_client=pg,
        milvus_client=milvus,
        llm_client=llm,
        cross_encoder=None,
        memory_manager=memory,
    )


def _construct(cls, config: dict[str, Any]):
    try:
        return cls(**config)
    except TypeError:
        return cls(config)
