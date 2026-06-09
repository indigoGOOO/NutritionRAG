"""KV提取器

从ContentChunk中利用LLM提取实体-属性键值对，
构造(key, value_jsonb)对用于写入PostgreSQL。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.indexing.llm_client import BaseLLMClient
from src.indexing.models import ContentChunk, DocCategory, KVPair

logger = logging.getLogger(__name__)

# 提取prompt模板
EXTRACTION_SYSTEM_PROMPT = """你是一个营养学知识提取专家。你的任务是从给定文本中提取结构化的实体信息。
请严格按照JSON格式输出，不要包含任何其他文字。"""

EXTRACTION_PROMPT_TEMPLATE = """从以下文本中提取所有食材、菜品、营养素等实体及其属性。

文本内容：
{text}

文档类型：{doc_type}

请提取为以下JSON格式：
{{
  "entities": [
    {{
      "name": "实体名称",
      "type": "ingredient|dish|nutrient|symptom|person",
      "attributes": {{
        "category": "分类",
        "description": "简要描述",
        "nutrition_tags": ["标签1", "标签2"],
        "contraindications": ["禁忌1"],
        "suitable_for": ["适宜人群"],
        "related_entities": ["关联实体名"]
      }}
    }}
  ]
}}

注意：
- 只提取文本中明确提到的实体
- attributes中只填写文本中有依据的字段，没有的留空列表或空字符串
- name字段使用标准化名称（如"西红柿"而非"番茄"）"""


class KVExtractor:
    """KV键值对提取器"""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def extract(self, chunks: list[ContentChunk]) -> list[KVPair]:
        """从chunks中批量提取KV对"""
        all_kv_pairs = []

        for chunk in chunks:
            try:
                pairs = self._extract_from_chunk(chunk)
                all_kv_pairs.extend(pairs)
            except Exception as e:
                logger.warning(f"KV提取失败 chunk={chunk.chunk_id[:8]}: {e}")

        # 合并同名实体
        merged = self._merge_duplicates(all_kv_pairs)
        logger.info(f"KV提取完成: {len(chunks)} chunks → {len(merged)} KV pairs")
        return merged

    def _extract_from_chunk(self, chunk: ContentChunk) -> list[KVPair]:
        """从单个chunk提取KV对"""
        if not chunk.content.strip():
            return []

        # 短文本跳过LLM，用规则提取
        if len(chunk.content) < 50:
            return self._rule_based_extract(chunk)

        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            text=chunk.content[:2000],  # 限制输入长度
            doc_type=chunk.doc_category.value,
        )

        try:
            result = self.llm.extract_structured(
                prompt=prompt,
                schema={"entities": []},
                system=EXTRACTION_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.debug(f"LLM提取失败，回退到规则提取: {e}")
            return self._rule_based_extract(chunk)

        return self._parse_extraction_result(result, chunk)

    def _parse_extraction_result(self, result: dict, chunk: ContentChunk) -> list[KVPair]:
        """解析LLM提取结果为KVPair列表"""
        pairs = []
        entities = result.get("entities", [])

        for entity in entities:
            name = entity.get("name", "").strip()
            if not name:
                continue

            pair = KVPair(
                key=name,
                value={
                    "type": entity.get("type", "unknown"),
                    "attributes": entity.get("attributes", {}),
                    "source_doc_type": chunk.doc_category.value,
                },
                entity_type=entity.get("type", "unknown"),
                source_chunk_id=chunk.chunk_id,
                source_doc_id=chunk.source_doc_id,
            )
            pairs.append(pair)

        return pairs

    def _rule_based_extract(self, chunk: ContentChunk) -> list[KVPair]:
        """基于规则的简单提取（用于短文本或LLM失败时的fallback）"""
        pairs = []

        if chunk.doc_category == DocCategory.NUTRITION:
            pairs.extend(self._extract_nutrition_kv(chunk))
        elif chunk.doc_category == DocCategory.RECIPE:
            pairs.extend(self._extract_recipe_kv(chunk))

        return pairs

    def _extract_nutrition_kv(self, chunk: ContentChunk) -> list[KVPair]:
        """从营养成分文本中规则提取"""
        import re

        pairs = []
        # 匹配 "营养素名 数值 单位" 模式
        pattern = re.compile(r"([一-鿿]+[A-Za-z]*\d*)\s*[:：]?\s*([\d.]+)\s*(g|mg|μg|kcal|kJ|%)?")
        for match in pattern.finditer(chunk.content):
            name = match.group(1).strip()
            value = match.group(2)
            unit = match.group(3) or ""
            pairs.append(KVPair(
                key=name,
                value={"amount": value, "unit": unit, "type": "nutrient"},
                entity_type="nutrient",
                source_chunk_id=chunk.chunk_id,
                source_doc_id=chunk.source_doc_id,
            ))

        return pairs

    def _extract_recipe_kv(self, chunk: ContentChunk) -> list[KVPair]:
        """从菜谱文本中规则提取"""
        pairs = []
        # 如果是配料chunk，提取食材名
        if "recipe_ingredients" in chunk.chunk_type:
            lines = chunk.content.split("\n")
            for line in lines:
                line = line.strip().lstrip("- ")
                if line and len(line) < 30:
                    parts = line.split()
                    if parts:
                        name = parts[0]
                        amount = " ".join(parts[1:]) if len(parts) > 1 else ""
                        pairs.append(KVPair(
                            key=name,
                            value={"amount": amount, "type": "ingredient", "role": "配料"},
                            entity_type="ingredient",
                            source_chunk_id=chunk.chunk_id,
                            source_doc_id=chunk.source_doc_id,
                        ))
        return pairs

    @staticmethod
    def _merge_duplicates(pairs: list[KVPair]) -> list[KVPair]:
        """合并同名实体的KV对"""
        merged: dict[str, KVPair] = {}

        for pair in pairs:
            if pair.key in merged:
                existing = merged[pair.key]
                # 合并attributes
                existing_attrs = existing.value.get("attributes", {})
                new_attrs = pair.value.get("attributes", {})
                for k, v in new_attrs.items():
                    if v and not existing_attrs.get(k):
                        existing_attrs[k] = v
                existing.value["attributes"] = existing_attrs
            else:
                merged[pair.key] = pair

        return list(merged.values())
