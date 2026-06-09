"""Models for user-owned saved content."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.indexing.models import DocCategory


class UserContentType(str, Enum):
    RECIPE = "recipe"
    MEAL_PLAN = "meal_plan"
    WORKOUT_PLAN = "workout_plan"
    FOOD_LOG = "food_log"
    BODY_METRICS = "body_metrics"
    LAB_REPORT = "lab_report"


CONTENT_TYPE_TO_DOC_CATEGORY = {
    UserContentType.RECIPE: DocCategory.RECIPE,
    UserContentType.MEAL_PLAN: DocCategory.DAILY,
    UserContentType.WORKOUT_PLAN: DocCategory.PERSONAL,
    UserContentType.FOOD_LOG: DocCategory.DAILY,
    UserContentType.BODY_METRICS: DocCategory.PERSONAL,
    UserContentType.LAB_REPORT: DocCategory.MEDICAL,
}


@dataclass
class UserContentClassification:
    content_type: UserContentType | None
    confidence: float
    reason: str
    signals: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_type": self.content_type.value if self.content_type else "",
            "confidence": self.confidence,
            "reason": self.reason,
            "signals": self.signals,
        }

