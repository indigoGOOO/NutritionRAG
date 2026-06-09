"""Improved KV extractor with LLM EAV extraction and deterministic fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.indexing.llm_client import BaseLLMClient
from src.indexing.models import ContentChunk, KVPair

logger = logging.getLogger(__name__)

ENTITY_RECOGNITION_PROMPT = """从以下文本中识别所有值得结构化的实体。

文本内容：
{text}

文档类型：{doc_type}

实体类型：
- ingredient: 食材，例如鸡蛋、番茄、牛奶
- nutrient: 营养素，例如蛋白质、热量、脂肪
- dish: 菜品，例如番茄炒蛋
- symptom: 症状/疾病，例如糖尿病、高血压
- person: 人群，例如孕妇、儿童
- cooking_method: 烹饪方法，例如炒、蒸、烤

请返回 JSON：
{{
  "entities": [
    {{
      "name": "实体名称",
      "type": "entity_type",
      "context": "实体在文本中的上下文"
    }}
  ]
}}

只识别文本中明确提到的实体，不要推断。"""

EAV_GENERATION_PROMPT = """为以下实体生成标准化属性-值对，使用 EAV 格式。

实体名称：{entity_name}
实体类型：{entity_type}
上下文：{context}

请只生成文本中有依据的属性。常见属性包括：
- ingredient: category, calories, protein, fat, carbs, suitable_for, contraindications
- nutrient: category, reference_value, unit, sources
- dish: category, main_ingredients, cooking_method, difficulty, time_required, calories
- symptom: category, forbidden_ingredients, recommended_ingredients, dietary_advice

请返回 JSON：
{{
  "entity_id": "entity_001",
  "entity_name": "{entity_name}",
  "entity_type": "{entity_type}",
  "attributes": [
    {{
      "attr": "属性名",
      "value": "属性值",
      "confidence": 0.0,
      "source": "信息来源"
    }}
  ]
}}"""


class KVExtractor:
    """Two-stage EAV extractor.

    First tries LLM-based entity and attribute extraction. If no usable KV is
    returned, falls back to deterministic nutrition regex extraction so the
    indexing pipeline still produces basic structured facts.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def extract(self, chunks: list[ContentChunk]) -> list[KVPair]:
        all_kv_pairs: list[KVPair] = []

        for chunk in chunks:
            try:
                chunk_pairs = self._extract_with_llm(chunk)
                if not chunk_pairs:
                    chunk_pairs = self._extract_with_rules(chunk)
                all_kv_pairs.extend(chunk_pairs)
            except Exception as exc:
                logger.warning(f"KV提取失败 chunk={chunk.chunk_id[:8]}: {exc}")

        merged = self._merge_duplicates(all_kv_pairs)
        logger.info(f"KV提取完成: {len(chunks)} chunks -> {len(merged)} KV pairs")
        return merged

    def _extract_with_llm(self, chunk: ContentChunk) -> list[KVPair]:
        entities = self._recognize_entities(chunk)
        pairs = []
        for entity in entities:
            eav = self._generate_eav(entity, chunk)
            if eav and eav.get("attributes"):
                pairs.append(self._eav_to_kvpair(eav, chunk))
        return pairs

    def _recognize_entities(self, chunk: ContentChunk) -> list[dict[str, Any]]:
        if not chunk.content.strip() or len(chunk.content) < 30:
            return []

        prompt = ENTITY_RECOGNITION_PROMPT.format(
            text=chunk.content[:2000],
            doc_type=chunk.doc_category.value,
        )
        try:
            result = self.llm.extract_structured(prompt=prompt, schema={"entities": []})
            entities = result.get("entities", [])
            return [e for e in entities if isinstance(e, dict) and e.get("name")]
        except Exception as exc:
            logger.debug(f"实体识别失败: {exc}")
            return []

    def _generate_eav(self, entity: dict[str, Any], chunk: ContentChunk) -> dict[str, Any] | None:
        entity_name = str(entity.get("name", "")).strip()
        entity_type = str(entity.get("type", "unknown")).strip() or "unknown"
        context = str(entity.get("context", "")).strip() or chunk.content[:500]
        if not entity_name:
            return None

        prompt = EAV_GENERATION_PROMPT.format(
            entity_name=entity_name,
            entity_type=entity_type,
            context=context,
        )
        try:
            return self.llm.extract_structured(
                prompt=prompt,
                schema={
                    "entity_id": "string",
                    "entity_name": "string",
                    "entity_type": "string",
                    "attributes": [],
                },
            )
        except Exception as exc:
            logger.debug(f"EAV生成失败 {entity_name}: {exc}")
            return None

    def _extract_with_rules(self, chunk: ContentChunk) -> list[KVPair]:
        text = chunk.content
        entity_name = self._guess_primary_entity(text)
        attributes = self._extract_nutrition_attributes(text)
        if not entity_name or not attributes:
            return []

        return [
            KVPair(
                key=entity_name,
                value={
                    "entity_id": f"rule_{entity_name}",
                    "entity_type": "ingredient",
                    "attributes": attributes,
                    "source_doc_type": chunk.doc_category.value,
                    "extraction_method": "rule_fallback",
                },
                entity_type="ingredient",
                source_chunk_id=chunk.chunk_id,
                source_doc_id=chunk.source_doc_id,
            )
        ]

    @staticmethod
    def _guess_primary_entity(text: str) -> str:
        patterns = [
            r"([\u4e00-\u9fffA-Za-z]{1,20})营养",
            r"([\u4e00-\u9fffA-Za-z]{1,20})每\s*100\s*克",
            r"([\u4e00-\u9fffA-Za-z]{1,20})含",
        ]
        stopwords = {"每", "适合", "一般", "高蛋白", "低碳水"}
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip("，。；:：、 ")
                if name and name not in stopwords:
                    return name
        return ""

    @staticmethod
    def _extract_nutrition_attributes(text: str) -> list[dict[str, Any]]:
        attr_patterns = {
            "protein": r"蛋白质\s*([0-9]+(?:\.[0-9]+)?)\s*(克|g)",
            "fat": r"脂肪\s*([0-9]+(?:\.[0-9]+)?)\s*(克|g)",
            "carbs": r"(?:碳水化合物|碳水)\s*([0-9]+(?:\.[0-9]+)?)\s*(克|g)",
            "calories": r"(?:热量|能量)\s*([0-9]+(?:\.[0-9]+)?)\s*(千卡|kcal|大卡)",
        }
        attributes = []
        for attr, pattern in attr_patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                attributes.append({
                    "attr": attr,
                    "value": f"{match.group(1)}{match.group(2)}",
                    "confidence": 0.9,
                    "source": "rule_fallback",
                })

        qualitative_rules = [
            ("suitable_for", r"适合([^。；\n]+)"),
            ("dietary_feature", r"(高蛋白|低碳水|低脂|低糖|低盐)"),
        ]
        for attr, pattern in qualitative_rules:
            for match in re.finditer(pattern, text):
                attributes.append({
                    "attr": attr,
                    "value": match.group(1).strip(),
                    "confidence": 0.75,
                    "source": "rule_fallback",
                })
        return attributes

    @staticmethod
    def _eav_to_kvpair(eav: dict[str, Any], chunk: ContentChunk) -> KVPair:
        return KVPair(
            key=eav.get("entity_name", ""),
            value={
                "entity_id": eav.get("entity_id", ""),
                "entity_type": eav.get("entity_type", "unknown"),
                "attributes": eav.get("attributes", []),
                "source_doc_type": chunk.doc_category.value,
                "extraction_method": "llm",
            },
            entity_type=eav.get("entity_type", "unknown"),
            source_chunk_id=chunk.chunk_id,
            source_doc_id=chunk.source_doc_id,
        )

    @staticmethod
    def _merge_duplicates(pairs: list[KVPair]) -> list[KVPair]:
        merged: dict[str, KVPair] = {}
        for pair in pairs:
            if pair.key not in merged:
                merged[pair.key] = pair
                continue

            existing_attrs = merged[pair.key].value.get("attributes", [])
            new_attrs = pair.value.get("attributes", [])
            attr_dict = {attr.get("attr"): attr for attr in existing_attrs}
            for new_attr in new_attrs:
                attr_name = new_attr.get("attr")
                if attr_name not in attr_dict:
                    attr_dict[attr_name] = new_attr
                elif new_attr.get("confidence", 0) > attr_dict[attr_name].get("confidence", 0):
                    attr_dict[attr_name] = new_attr
            merged[pair.key].value["attributes"] = list(attr_dict.values())

        return list(merged.values())
