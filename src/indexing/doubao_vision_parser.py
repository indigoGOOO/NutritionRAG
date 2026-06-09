"""Doubao Vision parser for natural food-related images.

For ambiguous images, parse_with_classification performs image classification
and content parsing in a single vision-model call.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Literal

from src.indexing.llm_client import BaseLLMClient
from src.indexing.models import (
    BlockMetadata,
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    SourceType,
    UnifiedDocument,
)

logger = logging.getLogger(__name__)

VisionType = Literal["document", "natural"]


class DoubaoVisionParser:
    """Vision parser for food photos, meals, fridge images, and takeout photos."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def parse(self, image_path: Path, image_type: str = "food_photo") -> UnifiedDocument:
        """Parse a known natural image type."""
        logger.info("Doubao Vision parse: %s (type=%s)", image_path.name, image_type)

        image_base64 = self._encode_image(image_path)
        image_mime = self._guess_mime_type(image_path)
        description = self._analyze_image(image_base64, image_type, image_mime)
        return self._to_document(
            image_path=image_path,
            description=description,
            image_type=image_type,
            confidence=1.0,
            classification_reason="provided_by_router",
        )

    def parse_with_classification(self, image_path: Path) -> tuple[VisionType, str, float, UnifiedDocument | None]:
        """Classify and parse the image in one vision call.

        Returns:
            (image_kind, subtype, confidence, document)
            document is only populated for natural images. If the model decides
            the image is document-like, caller should route it to Docling.
        """
        logger.info("Doubao Vision classify+parse: %s", image_path.name)
        image_base64 = self._encode_image(image_path)
        image_mime = self._guess_mime_type(image_path)

        try:
            raw = self.llm.generate_with_image(
                prompt=self._classification_parse_prompt(),
                image_base64=image_base64,
                image_mime=image_mime,
                system=(
                    "你是营养学专家和图片理解助手。请判断图片类型，并在自然食物场景下"
                    "给出可用于膳食索引的结构化中文描述。"
                ),
            )
            result = self._parse_json(raw)
        except Exception as exc:
            logger.error("Doubao Vision classify+parse failed: %s", exc)
            description = f"图片分析失败: {exc}"
            doc = self._to_document(
                image_path=image_path,
                description=description,
                image_type="food_photo",
                confidence=0.0,
                classification_reason="vision_call_failed",
            )
            return "natural", "food_photo", 0.0, doc

        image_kind = str(result.get("type", "natural")).strip().lower()
        if image_kind not in {"document", "natural"}:
            image_kind = "natural"

        subtype = str(result.get("subtype", "food_photo") or "food_photo")
        confidence = self._clamp_float(result.get("confidence", 0.5), default=0.5)
        reason = str(result.get("reason", "") or "")

        if image_kind == "document":
            return "document", subtype, confidence, None

        description = self._description_from_result(result)
        doc = self._to_document(
            image_path=image_path,
            description=description,
            image_type=subtype,
            confidence=confidence,
            classification_reason=reason,
            raw_result=result,
        )
        return "natural", subtype, confidence, doc

    def _analyze_image(self, image_base64: str, image_type: str, image_mime: str) -> str:
        prompt = self._analysis_prompt(image_type)
        try:
            return self.llm.generate_with_image(
                prompt=prompt,
                image_base64=image_base64,
                image_mime=image_mime,
                system="你是营养学专家和食物识别专家。请准确、详细地分析图片中的食物信息。",
            )
        except Exception as exc:
            logger.error("Doubao Vision analysis failed: %s", exc)
            return f"图片分析失败: {exc}"

    @staticmethod
    def _classification_parse_prompt() -> str:
        return """请分析这张图片，并返回严格 JSON。

如果图片是文档型图片（营养成分表、包装标签、菜谱截图、医学报告、PDF/扫描页），返回 type=document。
如果图片是自然场景图片（食物照片、餐盘、外卖、冰箱食材），返回 type=natural，并填写 description。

JSON 字段：
{
  "type": "document 或 natural",
  "subtype": "nutrition_table|recipe|medical_report|pdf_page|food_photo|meal|fridge|takeout",
  "confidence": 0.0,
  "reason": "分类理由",
  "description": "自然场景图片的结构化中文描述；document 时可为空",
  "foods": ["可见食物或食材"],
  "nutrition_notes": "营养特征、烹饪方式、份量估计和饮食建议"
}

要求：
- 只输出 JSON，不要 Markdown。
- 对自然图片，description 应适合后续分块、向量化和检索。
- 不确定具体食物时说明不确定，不要编造。
"""

    @staticmethod
    def _analysis_prompt(image_type: str) -> str:
        prompts = {
            "food_photo": """请详细描述这张食物照片：
1. 主要食材和菜品名称
2. 可见的烹饪方式
3. 估计份量
4. 营养特征
5. 适合或需要谨慎的人群

请用结构化中文组织信息。""",
            "meal": """请分析这份饭菜：
1. 主食
2. 主菜
3. 配菜
4. 汤或饮品
5. 整体营养搭配评价
6. 热量估计（低/中/高）

请用清晰的列表格式。""",
            "fridge": """请列出冰箱中可见的食材：
1. 蔬菜类
2. 肉类/蛋白质
3. 乳制品
4. 调味料
5. 其他食材

对每个食材简要描述状态和用途。""",
            "takeout": """请分析这份外卖：
1. 餐厅/菜系类型
2. 主要菜品
3. 包装信息
4. 份量估计
5. 营养特征
6. 食用建议

请用结构化格式。""",
        }
        return prompts.get(image_type, prompts["food_photo"])

    @staticmethod
    def _description_from_result(result: dict[str, Any]) -> str:
        description = str(result.get("description", "") or "").strip()
        foods = result.get("foods", [])
        notes = str(result.get("nutrition_notes", "") or "").strip()

        parts = []
        if description:
            parts.append(description)
        if isinstance(foods, list) and foods:
            parts.append("可见食物/食材：" + "、".join(str(item) for item in foods if item))
        if notes:
            parts.append("营养与饮食备注：" + notes)
        return "\n\n".join(parts) if parts else "图片中包含食物或餐食，但模型未能给出详细描述。"

    def _to_document(
        self,
        *,
        image_path: Path,
        description: str,
        image_type: str,
        confidence: float,
        classification_reason: str,
        raw_result: dict[str, Any] | None = None,
    ) -> UnifiedDocument:
        block = DocumentBlock(
            block_type=BlockType.TEXT,
            content=description,
            metadata=BlockMetadata(position=0, confidence=confidence),
        )
        metadata = DocumentMetadata(
            source_path=str(image_path),
            source_type=SourceType.IMAGE,
            title=self._extract_title(image_type),
            file_size_bytes=image_path.stat().st_size,
            extra={
                "image_type": image_type,
                "image_kind": "natural",
                "classification_confidence": confidence,
                "classification_reason": classification_reason,
                "vision_model": "doubao",
                "single_vision_call": raw_result is not None,
            },
        )
        if raw_result is not None:
            metadata.extra["vision_result"] = raw_result
        return UnifiedDocument(blocks=[block], metadata=metadata)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            end_idx = next((i for i, line in enumerate(lines) if line.strip() == "```"), len(lines))
            text = "\n".join(lines[:end_idx]).strip()
        return json.loads(text)

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _guess_mime_type(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        return mime_type or "image/jpeg"

    @staticmethod
    def _extract_title(image_type: str) -> str:
        titles = {
            "food_photo": "食物照片分析",
            "meal": "饭菜分析",
            "fridge": "冰箱食材清单",
            "takeout": "外卖分析",
        }
        return titles.get(image_type, "图片分析")

    @staticmethod
    def _clamp_float(value: Any, default: float = 0.5) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return max(0.0, min(1.0, numeric))
