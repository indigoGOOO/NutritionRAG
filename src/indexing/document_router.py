"""Document router.

Routes supported input files to parsers and returns UnifiedDocument.
Image routing is optimized to avoid two vision-model calls for natural images:
document-like images use Docling directly, while ambiguous/natural images use
one Doubao Vision classify+parse call.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import SUPPORTED_EXTENSIONS
from src.indexing.models import UnifiedDocument

logger = logging.getLogger(__name__)


class DocumentRouter:
    """Dispatch files to the proper parser."""

    def __init__(self, llm_client=None):
        from src.indexing.docling_parser import DoclingParser
        from src.indexing.doubao_vision_parser import DoubaoVisionParser
        from src.indexing.image_classifier import ImageClassifier
        from src.indexing.text_parser import TextParser

        self.docling_parser = DoclingParser()
        self.text_parser = TextParser()
        self.image_classifier = ImageClassifier()
        self.doubao_vision_parser = DoubaoVisionParser(llm_client=llm_client) if llm_client else None

    def route(self, file_path: Path) -> UnifiedDocument:
        """Route a file based on extension."""
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        suffix = file_path.suffix.lower()
        file_type = self._detect_type(suffix)

        logger.info("Route document: %s -> %s", file_path.name, file_type)

        if file_type == "pdf":
            return self.docling_parser.parse(file_path)
        if file_type == "image":
            return self._route_image(file_path)
        if file_type == "text":
            return self.text_parser.parse(file_path)
        raise ValueError(f"Unsupported file type: {suffix} ({file_path.name})")

    def _route_image(self, image_path: Path) -> UnifiedDocument:
        """Route image files with at most one vision-model call for natural images."""
        route_decision = self.image_classifier.decide_image_route(image_path)
        routing_trace = {"lightweight": route_decision.to_trace()}
        logger.info(
            "Lightweight image route: %s subtype=%s decision=%s confidence=%.2f",
            route_decision.kind,
            route_decision.subtype,
            route_decision.decision,
            route_decision.confidence,
        )

        if route_decision.decision == "direct_document":
            document = self.docling_parser.parse(image_path)
            return self._attach_routing_trace(document, routing_trace)

        if self.doubao_vision_parser:
            vision_kind, vision_subtype, vision_confidence, document = (
                self.doubao_vision_parser.parse_with_classification(image_path)
            )
            routing_trace["vision"] = {
                "kind": vision_kind,
                "subtype": vision_subtype,
                "confidence": vision_confidence,
            }
            logger.info(
                "Vision image route: %s subtype=%s confidence=%.2f",
                vision_kind,
                vision_subtype,
                vision_confidence,
            )
            if vision_kind == "document":
                parsed = self.docling_parser.parse(image_path)
                return self._attach_routing_trace(parsed, routing_trace)
            if document is not None:
                return self._attach_routing_trace(document, routing_trace)

        if route_decision.kind == "document":
            document = self.docling_parser.parse(image_path)
            return self._attach_routing_trace(document, routing_trace)

        if self.doubao_vision_parser:
            document = self.doubao_vision_parser.parse(image_path, image_type=route_decision.subtype)
            return self._attach_routing_trace(document, routing_trace)

        logger.warning("Vision parser unavailable, falling back to Docling for image: %s", image_path.name)
        document = self.docling_parser.parse(image_path)
        return self._attach_routing_trace(document, routing_trace)

    def route_text(self, text: str, source_name: str = "user_input") -> UnifiedDocument:
        """Route raw text input."""
        return self.text_parser.parse_from_string(text, source_name=source_name)

    def route_batch(self, directory: Path, recursive: bool = True) -> list[UnifiedDocument]:
        """Route all supported files under a directory."""
        documents = []
        pattern = "**/*" if recursive else "*"

        for file_path in sorted(directory.glob(pattern)):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if self._detect_type(suffix) is None:
                continue
            try:
                documents.append(self.route(file_path))
            except Exception as exc:
                logger.error("Parse failed %s: %s", file_path.name, exc)

        logger.info("Batch parse complete: %s documents", len(documents))
        return documents

    @staticmethod
    def _detect_type(suffix: str) -> str | None:
        for file_type, extensions in SUPPORTED_EXTENSIONS.items():
            if suffix in extensions:
                return file_type
        return None

    @staticmethod
    def _attach_routing_trace(document: UnifiedDocument, routing_trace: dict) -> UnifiedDocument:
        document.metadata.extra["routing_trace"] = routing_trace
        return document
