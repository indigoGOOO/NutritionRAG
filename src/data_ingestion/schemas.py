"""Canonical schemas used by bulk data importers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SourceMetadata:
    source: str
    source_type: str
    source_url: str = ""
    language: str = ""
    confidence: float = 0.8
    updated_at: str = field(default_factory=lambda: datetime.utcnow().date().isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "source": self.source,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "language": self.language,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
        }
        data.update(self.extra)
        return data


@dataclass
class FoodEntity:
    entity_id: str
    name: str
    entity_type: str = "food"
    aliases: list[str] = field(default_factory=list)
    source: SourceMetadata | None = None


@dataclass
class NutrientProfile:
    entity_id: str
    entity_type: str = "food"
    nutrients: dict[str, Any] = field(default_factory=dict)
    serving_basis: str = "per_100g"
    source: SourceMetadata | None = None


@dataclass
class RelationFact:
    subject: str
    predicate: str
    object: str
    subject_type: str = "unknown"
    object_type: str = "unknown"
    confidence: float = 0.8
    source: SourceMetadata | None = None


@dataclass
class TextDocument:
    doc_id: str
    content: str
    title: str = ""
    doc_category: str = "nutrition"
    source: SourceMetadata | None = None


@dataclass
class ImportBatch:
    entities: list[FoodEntity] = field(default_factory=list)
    nutrient_profiles: list[NutrientProfile] = field(default_factory=list)
    relations: list[RelationFact] = field(default_factory=list)
    documents: list[TextDocument] = field(default_factory=list)

