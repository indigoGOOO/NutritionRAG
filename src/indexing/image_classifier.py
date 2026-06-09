"""Lightweight image classifier for document-like vs natural images.

The score here is not a calibrated probability. The router should prefer the
explicit decision field:
- direct_document: strong document evidence, route directly to Docling
- vision_required: weak/ambiguous/natural evidence, let Vision decide once
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

ImageKind = Literal["document", "natural", "unknown"]
RouteDecision = Literal["direct_document", "vision_required"]


@dataclass
class ImageRouteDecision:
    kind: ImageKind
    subtype: str
    confidence: float
    decision: RouteDecision
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)

    def to_trace(self) -> dict[str, Any]:
        return asdict(self)


class ImageClassifier:
    """Cheap pre-router before expensive vision parsing."""

    DOCUMENT_FILENAME_KEYWORDS = {
        "nutrition_table": [
            "nutrition",
            "营养",
            "成分",
            "label",
            "标签",
            "table",
            "report",
            "报告",
            "scan",
            "扫描",
            "pdf",
            "page",
            "receipt",
            "票据",
        ],
        "recipe": ["recipe", "菜谱", "食谱", "步骤", "做法"],
        "medical_report": ["medical", "体检", "化验", "检查", "诊断", "医院"],
    }

    NATURAL_FILENAME_KEYWORDS = {
        "food_photo": ["food", "meal", "dish", "餐", "饭", "菜", "食物", "照片"],
        "fridge": ["fridge", "冰箱", "冷藏"],
        "takeout": ["takeout", "外卖", "餐盒"],
    }

    OCR_DOCUMENT_KEYWORDS = {
        "nutrition_table": [
            "营养成分",
            "营养成份",
            "蛋白质",
            "脂肪",
            "碳水",
            "碳水化合物",
            "能量",
            "热量",
            "钠",
            "每100",
            "nrv",
            "配料",
            "nutrition",
            "protein",
            "fat",
            "carbohydrate",
            "sodium",
            "ingredients",
        ],
        "recipe": ["食谱", "菜谱", "配料", "步骤", "做法", "烹饪", "recipe", "ingredients"],
        "medical_report": ["报告", "诊断", "检验", "检查", "医院", "医生", "medical", "diagnosis"],
        "pdf_page": ["目录", "摘要", "参考文献", "扫描", "page", "abstract"],
    }

    MIN_OCR_CHARS_FOR_DOCUMENT = 18

    def __init__(self, llm_client=None, enable_ocr: bool = True):
        # llm_client is kept for backwards compatibility; this class never calls LLM.
        self.llm = llm_client
        self.enable_ocr = enable_ocr

    def classify(self, image_path: Path) -> tuple[Literal["document", "natural"], str, float]:
        """Backward-compatible tuple interface."""
        decision = self.decide_image_route(image_path)
        if decision.kind == "unknown":
            return "natural", "food_photo", decision.confidence
        return decision.kind, decision.subtype, decision.confidence

    def classify_lightweight(self, image_path: Path) -> tuple[ImageKind, str, float]:
        """Backward-compatible tuple interface for older tests/callers."""
        decision = self.decide_image_route(image_path)
        return decision.kind, decision.subtype, decision.confidence

    def decide_image_route(self, image_path: Path) -> ImageRouteDecision:
        """Return a route decision with traceable signals."""
        filename = image_path.name.lower()

        doc_subtype, doc_hits = self._keyword_hits(filename, self.DOCUMENT_FILENAME_KEYWORDS)
        if doc_hits:
            return ImageRouteDecision(
                kind="document",
                subtype=doc_subtype,
                confidence=0.9,
                decision="direct_document",
                reason="filename_strong_document_match",
                signals={"filename_hits": doc_hits},
            )

        ocr_decision = self._classify_with_ocr_probe(image_path)
        if ocr_decision:
            return ocr_decision

        nat_subtype, nat_hits = self._keyword_hits(filename, self.NATURAL_FILENAME_KEYWORDS)
        if nat_hits:
            return ImageRouteDecision(
                kind="natural",
                subtype=nat_subtype,
                confidence=0.78,
                decision="vision_required",
                reason="filename_natural_hint",
                signals={"filename_hits": nat_hits},
            )

        try:
            file_size = image_path.stat().st_size
        except OSError:
            logger.debug("Unable to stat image, treating it as ambiguous: %s", image_path)
            return ImageRouteDecision(
                kind="unknown",
                subtype="unknown",
                confidence=0.0,
                decision="vision_required",
                reason="stat_failed",
            )

        if file_size < 80 * 1024:
            return ImageRouteDecision(
                kind="document",
                subtype="pdf_page",
                confidence=0.55,
                decision="vision_required",
                reason="weak_small_file_document_hint",
                signals={"file_size_bytes": file_size},
            )

        return ImageRouteDecision(
            kind="unknown",
            subtype="unknown",
            confidence=0.35,
            decision="vision_required",
            reason="no_strong_lightweight_signal",
            signals={"file_size_bytes": file_size},
        )

    def _classify_with_ocr_probe(self, image_path: Path) -> ImageRouteDecision | None:
        if not self.enable_ocr:
            return None

        text = self._extract_ocr_text(image_path)
        normalized = self._normalize_ocr_text(text)
        if not normalized:
            return None

        best_subtype = "pdf_page"
        best_hits: list[str] = []
        for subtype, keywords in self.OCR_DOCUMENT_KEYWORDS.items():
            hits = [keyword for keyword in keywords if keyword.lower() in normalized]
            if len(hits) > len(best_hits):
                best_subtype = subtype
                best_hits = hits

        readable_chars = len(re.sub(r"\s+", "", normalized))
        signals = {
            "ocr_hit_count": len(best_hits),
            "ocr_hits": best_hits[:8],
            "ocr_char_count": readable_chars,
            "ocr_text_preview": normalized[:160],
        }
        if not best_hits and readable_chars < self.MIN_OCR_CHARS_FOR_DOCUMENT:
            return None

        confidence = 0.58
        if readable_chars >= self.MIN_OCR_CHARS_FOR_DOCUMENT:
            confidence += 0.12
        confidence += min(len(best_hits), 5) * 0.06
        if best_subtype in {"nutrition_table", "medical_report"} and len(best_hits) >= 2:
            confidence += 0.08
        confidence = min(confidence, 0.93)

        if self._is_strong_ocr_document(best_subtype, best_hits, readable_chars):
            return ImageRouteDecision(
                kind="document",
                subtype=best_subtype,
                confidence=confidence,
                decision="direct_document",
                reason="ocr_strong_document_match",
                signals=signals,
            )

        return ImageRouteDecision(
            kind="document",
            subtype=best_subtype,
            confidence=confidence,
            decision="vision_required",
            reason="ocr_weak_document_hint",
            signals=signals,
        )

    @staticmethod
    def _is_strong_ocr_document(subtype: str, hits: list[str], readable_chars: int) -> bool:
        if readable_chars < ImageClassifier.MIN_OCR_CHARS_FOR_DOCUMENT:
            return False
        if subtype == "nutrition_table":
            return len(hits) >= 3
        if subtype == "medical_report":
            return len(hits) >= 2
        if subtype == "recipe":
            return len(hits) >= 3
        return len(hits) >= 3 and readable_chars >= 40

    def _extract_ocr_text(self, image_path: Path) -> str:
        """Best-effort OCR text extraction.

        OCR dependencies are optional. If Pillow, pytesseract, or the Tesseract
        binary is unavailable, this returns an empty string and routing continues.
        """
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            logger.debug("OCR dependencies unavailable; skip OCR probe")
            return ""

        try:
            with Image.open(image_path) as image:
                image = self._prepare_image_for_ocr(image)
                return pytesseract.image_to_string(image, lang="chi_sim+eng")
        except Exception as exc:
            logger.debug("OCR probe failed for %s: %s", image_path.name, exc)
            return ""

    @staticmethod
    def _prepare_image_for_ocr(image):
        image = image.convert("L")
        max_side = max(image.size)
        if max_side > 1800:
            scale = 1800 / max_side
            new_size = (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale)))
            image = image.resize(new_size)
        return image

    @staticmethod
    def _normalize_ocr_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip().lower()

    @staticmethod
    def _keyword_hits(text: str, keyword_map: dict[str, list[str]]) -> tuple[str, list[str]]:
        best_subtype = "unknown"
        best_hits: list[str] = []
        for subtype, keywords in keyword_map.items():
            hits = [keyword for keyword in keywords if keyword.lower() in text]
            if len(hits) > len(best_hits):
                best_subtype = subtype
                best_hits = hits
        return best_subtype, best_hits
