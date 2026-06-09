import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.content_classifier import ContentClassifier
from src.indexing.models import (
    BlockMetadata,
    BlockType,
    DocCategory,
    DocumentBlock,
    DocumentMetadata,
    UnifiedDocument,
)


def _doc(text: str, title: str = "", source_path: str = "") -> UnifiedDocument:
    return UnifiedDocument(
        blocks=[
            DocumentBlock(
                block_type=BlockType.TEXT,
                content=text,
                metadata=BlockMetadata(position=0),
            )
        ],
        metadata=DocumentMetadata(title=title, source_path=source_path),
    )


def test_classifier_writes_trace_to_document_metadata():
    classifier = ContentClassifier()
    document = _doc("每100g含热量144kcal，蛋白质13g，脂肪8g。")

    result_doc = classifier.classify_and_set(document)
    trace = result_doc.metadata.extra["classification"]

    assert result_doc.doc_category == DocCategory.NUTRITION
    assert trace["primary_category"] == "nutrition"
    assert trace["confidence"] > 0
    assert "keyword_hits" in trace["signals"]


def test_medical_strong_signal_wins():
    classifier = ContentClassifier()
    document = _doc("高血压患者饮食指南：应限制钠摄入，避免高盐腌制食品。")

    result = classifier.classify_with_trace(document)

    assert result.primary_category == DocCategory.MEDICAL
    assert result.method == "rule_strong"


class FakeLLM:
    def extract_structured(self, prompt, schema):
        return {
            "category": "recipe",
            "secondary_categories": ["nutrition"],
            "confidence": 0.88,
            "reason": "contains cooking instructions and ingredients",
        }


def test_llm_fallback_used_when_rule_confidence_low():
    classifier = ContentClassifier(llm_client=FakeLLM())
    document = _doc(
        "This document discusses food and meal preparation. "
        "It includes ingredients, nutrients, and general diet notes. " * 5
    )

    result = classifier.classify_with_trace(document)

    assert result.primary_category == DocCategory.RECIPE
    assert result.secondary_categories == [DocCategory.NUTRITION]
    assert result.method == "llm_fallback"

