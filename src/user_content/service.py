"""Service for saving user-owned content into the RAG stores."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from src.indexing.index_pipeline import IndexingPipeline
from src.indexing.models import PipelineResult
from src.storage.storage_manager import StorageManager
from src.user_content.classifier import UserContentClassifier
from src.user_content.models import CONTENT_TYPE_TO_DOC_CATEGORY, UserContentType


@dataclass
class SavedUserContent:
    saved: bool
    content_type: str
    title: str
    source_doc_id: str
    storage: dict[str, Any]
    classification: dict[str, Any]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "saved": self.saved,
            "content_type": self.content_type,
            "title": self.title,
            "source_doc_id": self.source_doc_id,
            "storage": self.storage,
            "classification": self.classification,
            "errors": self.errors,
        }


class UserContentService:
    """Save explicit user content such as recipes, plans, logs, and reports."""

    def __init__(
        self,
        pipeline: IndexingPipeline | None = None,
        storage: StorageManager | None = None,
        classifier: UserContentClassifier | None = None,
    ):
        self.pipeline = pipeline or IndexingPipeline()
        self.storage = storage or StorageManager()
        self.classifier = classifier or UserContentClassifier()

    def save_text(
        self,
        *,
        user_id: str,
        session_id: str,
        content: str,
        content_type: str | None = None,
        title: str = "",
        visibility: str = "private",
    ) -> SavedUserContent:
        classification = self.classifier.classify(content, explicit_type=content_type)
        if classification.content_type is None:
            return SavedUserContent(
                saved=False,
                content_type="",
                title=title,
                source_doc_id="",
                storage={},
                classification=classification.to_dict(),
                errors=["unsupported_or_unclear_user_content_type"],
            )

        content_type_value = classification.content_type.value
        safe_title = title or _default_title(classification.content_type)
        source_doc_id = _source_doc_id(user_id, content_type_value)
        result = self.pipeline.run_text(content, source_name=source_doc_id)
        self._annotate_result(
            result=result,
            user_id=user_id,
            session_id=session_id,
            content_type=classification.content_type,
            title=safe_title,
            source_doc_id=source_doc_id,
            visibility=visibility,
            classification=classification.to_dict(),
        )
        storage_result = self.storage.store_pipeline_result(result) if not result.errors else {}
        return SavedUserContent(
            saved=not bool(result.errors),
            content_type=content_type_value,
            title=safe_title,
            source_doc_id=source_doc_id,
            storage=storage_result,
            classification=classification.to_dict(),
            errors=result.errors,
        )

    def _annotate_result(
        self,
        *,
        result: PipelineResult,
        user_id: str,
        session_id: str,
        content_type: UserContentType,
        title: str,
        source_doc_id: str,
        visibility: str,
        classification: dict[str, Any],
    ) -> None:
        doc_category = CONTENT_TYPE_TO_DOC_CATEGORY[content_type]
        metadata = {
            "user_id": user_id,
            "session_id": session_id,
            "user_content_type": content_type.value,
            "source_type": "user_saved_content",
            "visibility": visibility,
            "title": title,
            "classification": classification,
        }
        result.doc_id = source_doc_id

        for chunk in result.chunks:
            chunk.doc_category = doc_category
            chunk.source_doc_id = source_doc_id
            chunk.metadata.update(metadata)

        for kv in result.kv_pairs:
            kv.source_doc_id = source_doc_id
            kv.entity_type = kv.entity_type or content_type.value
            if isinstance(kv.value, dict):
                kv.value.setdefault("metadata", {}).update(metadata)

        for triple in result.triples:
            triple.properties.update(metadata)


def _source_doc_id(user_id: str, content_type: str) -> str:
    return f"user_content:{user_id}:{content_type}:{uuid.uuid4().hex[:12]}"


def _default_title(content_type: UserContentType) -> str:
    return {
        UserContentType.RECIPE: "Saved recipe",
        UserContentType.MEAL_PLAN: "Saved meal plan",
        UserContentType.WORKOUT_PLAN: "Saved workout plan",
        UserContentType.FOOD_LOG: "Saved food log",
        UserContentType.BODY_METRICS: "Saved body metrics",
        UserContentType.LAB_REPORT: "Saved lab report",
    }[content_type]

