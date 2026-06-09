"""图谱构建器（优化版）

从KV提取器产生的EAV数据直接转为关系三元组。

优化要点（对比旧版）：
1. 利用EAV的attributes数组直接转换，无需额外LLM调用
2. 复杂属性（禁忌、适宜人群等）用规则提取关联实体
3. chunk级规则提取作为轻量补充（仅文本模式，不调LLM）
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.indexing.models import ContentChunk, GraphTriple, KVPair

logger = logging.getLogger(__name__)

# 复杂属性 → 关系类型映射
RELATION_ATTR_MAP: dict[str, str] = {
    "禁忌": "contraindicated_for",
    "contraindications": "contraindicated_for",
    "不宜": "contraindicated_for",
    "适宜人群": "suitable_for",
    "suitable_for": "suitable_for",
    "推荐食材": "recommends",
    "主要食材": "contains",
    "related_entities": "related_to",
    "烹饪方法": "cooked_by",
    "替代": "can_replace",
    "代替": "can_replace",
    "搭配": "pairs_with",
    "相克": "conflicts_with",
    "分类": "belongs_to",
    "category": "belongs_to",
    "来源": "source_of",
}

# 属性值中提取实体名称的正则
ENTITY_IN_VALUE_PATTERN = re.compile(r"([一-鿿]{2,6}(?:[、，,][一-鿿]{2,6})*)")


class GraphBuilder:
    """知识图谱关系构建器（优化版）"""

    # 文本模式 → 关系类型（用于chunk级规则补充）
    CONFLICT_PATTERN = re.compile(
        r"([一-鿿]{2,6})\s*[和与跟]\s*([一-鿿]{2,6})\s*(?:相克|不宜同食|不能一起|禁止搭配)"
    )
    REPLACE_PATTERN = re.compile(
        r"([一-鿿]{2,6})\s*(?:可以)?\s*(?:替代|代替|换成)\s*([一-鿿]{2,6})"
    )

    def __init__(self, llm_client=None):
        # 优化后不需要LLM，保留参数兼容但不使用
        self.llm = llm_client

    def build(self, chunks: list[ContentChunk], kv_pairs: list[KVPair]) -> list[GraphTriple]:
        """从KV对的EAV数据构建关系三元组"""
        all_triples = []

        # 收集已知实体名列表，用于关系值中提取实体
        known_entities = {kv.key for kv in kv_pairs}

        # Phase 1: EAV → 三元组直接转换（核心，零LLM）
        eav_triples = self._extract_from_eav(kv_pairs, known_entities)
        all_triples.extend(eav_triples)

        # Phase 2: chunk级规则补充（轻量文本模式，不调LLM）
        for chunk in chunks:
            try:
                triples = self._rule_extract_from_chunk(chunk)
                all_triples.extend(triples)
            except Exception as e:
                logger.debug(f"chunk规则提取失败: {e}")

        # 去重
        unique_triples = self._deduplicate(all_triples)
        logger.info(
            f"图谱构建完成: {len(unique_triples)} 三元组 "
            f"(EAV直接转换: {len(eav_triples)}, "
            f"规则补充: {len(all_triples) - len(eav_triples)})"
        )
        return unique_triples

    def _extract_from_eav(
        self, kv_pairs: list[KVPair], known_entities: set[str]
    ) -> list[GraphTriple]:
        """
        Phase 1: EAV → 三元组直接转换

        每个KVPair代表一个实体，其value.attributes数组包含多个属性-值对。
        每个属性-值对可转为三元组：
        - 简单属性（热量、蛋白质等）：(entity, attr, value)
        - 复杂属性（禁忌、适宜人群等）：从value中提取关联实体后建关系
        """
        triples = []

        for kv in kv_pairs:
            entity_name = kv.key
            entity_type = kv.entity_type
            attrs = self._normalize_attributes(kv.value)

            for attr_item in attrs:
                attr_name = attr_item.get("attr", "")
                attr_value = attr_item.get("value", "")
                confidence = attr_item.get("confidence", 0.8)

                if not attr_name or not attr_value:
                    continue

                # 判断是否为复杂关系属性
                relation_type = self._match_relation_attr(attr_name)

                if relation_type:
                    # 复杂属性：从value中提取关联实体
                    related_entities = self._extract_entities_from_value(
                        attr_value, known_entities
                    )
                    if related_entities:
                        for rel_entity in related_entities:
                            triples.append(GraphTriple(
                                subject=entity_name,
                                subject_type=entity_type,
                                predicate=relation_type,
                                object=rel_entity,
                                object_type=self._infer_object_type(relation_type, entity_type),
                                properties={
                                    "confidence": confidence,
                                    "source_attr": attr_name,
                                },
                                source_chunk_id=kv.source_chunk_id,
                            ))
                    else:
                        # 没提取到已知实体 → 降级为描述性属性
                        triples.append(GraphTriple(
                            subject=entity_name,
                            subject_type=entity_type,
                            predicate=relation_type,
                            object=attr_value[:100],
                            object_type="description",
                            properties={
                                "confidence": confidence,
                                "source_attr": attr_name,
                                "is_attribute": True,
                            },
                            source_chunk_id=kv.source_chunk_id,
                        ))
                else:
                    # 简单属性：直接转为 (entity, attr, value) 三元组
                    obj_type = self._infer_literal_type(attr_name)
                    triples.append(GraphTriple(
                        subject=entity_name,
                        subject_type=entity_type,
                        predicate=attr_name,
                        object=attr_value if len(attr_value) < 100 else attr_value[:100],
                        object_type=obj_type,
                        properties={
                            "confidence": confidence,
                            "is_attribute": True,
                        },
                        source_chunk_id=kv.source_chunk_id,
                    ))

        return triples

    @staticmethod
    def _normalize_attributes(value: dict[str, Any]) -> list[dict[str, Any]]:
        """兼容不同KV提取器输出的属性结构。

        旧版/LLM提取器常返回 {"attributes": {"category": "..."}}
        优化版GraphBuilder期望 [{"attr": "...", "value": "..."}]。
        """
        if not isinstance(value, dict):
            return []

        attrs = value.get("attributes")
        if isinstance(attrs, list):
            return attrs

        normalized = []
        if isinstance(attrs, dict):
            for key, raw_value in attrs.items():
                if raw_value in ("", None, [], {}):
                    continue
                if isinstance(raw_value, list):
                    attr_value = "、".join(str(v) for v in raw_value if v)
                else:
                    attr_value = str(raw_value)
                normalized.append({
                    "attr": key,
                    "value": attr_value,
                    "confidence": value.get("confidence", 0.8),
                })

        for key in ("amount", "unit", "type", "role"):
            if key in value and value[key] not in ("", None):
                normalized.append({
                    "attr": key,
                    "value": str(value[key]),
                    "confidence": value.get("confidence", 0.8),
                })

        return normalized

    def _match_relation_attr(self, attr_name: str) -> str | None:
        """判断属性名是否对应关系类型"""
        for keyword, relation in RELATION_ATTR_MAP.items():
            if keyword in attr_name:
                return relation
        return None

    def _extract_entities_from_value(self, value: str, known_entities: set[str]) -> list[str]:
        """
        从属性值中提取关联实体名

        只从 known_entities 中精确匹配，避免产生脏数据。
        如"不宜与黄瓜同食"中有"黄瓜"且黄瓜在known_entities中则提取，
        否则返回空，由调用方降级为描述性属性。
        """
        extracted = []
        for entity in known_entities:
            if entity in value and entity not in extracted:
                extracted.append(entity)
        return extracted

    @staticmethod
    def _infer_object_type(relation: str, _subject_type: str) -> str:
        """为关系类型推断object_type"""
        type_map = {
            "contraindicated_for": "symptom",
            "suitable_for": "person",
            "recommends": "ingredient",
            "contains": "ingredient",
            "cooked_by": "cooking_method",
            "can_replace": "ingredient",
            "pairs_with": "ingredient",
            "conflicts_with": "ingredient",
            "belongs_to": "category",
            "source_of": "nutrient",
            "related_to": "unknown",
        }
        return type_map.get(relation, "unknown")

    @staticmethod
    def _infer_literal_type(attr_name: str) -> str:
        """为简单属性推断值的类型"""
        if any(k in attr_name for k in ["热量", "蛋白质", "脂肪", "碳水", "能量", "每"]):
            return "nutrient_value"
        elif any(k in attr_name for k in ["分类", "类别", "类型"]):
            return "category"
        elif any(k in attr_name for k in ["难度", "用时", "时间"]):
            return "description"
        elif any(k in attr_name for k in ["烹饪", "做法"]):
            return "cooking_method"
        else:
            return "property"

    def _rule_extract_from_chunk(self, chunk: ContentChunk) -> list[GraphTriple]:
        """
        Phase 2: chunk级规则补充（轻量，不调LLM）

        从文本中检测未在EAV中捕获的关系模式。
        作为EAV转换的补充，仅用正则。
        """
        triples = []
        text = chunk.content
        if not text or len(text) < 20:
            return triples

        # 检测相克关系：A和B相克
        for match in self.CONFLICT_PATTERN.finditer(text):
            triples.append(GraphTriple(
                subject=match.group(1),
                subject_type="ingredient",
                predicate="conflicts_with",
                object=match.group(2),
                object_type="ingredient",
                properties={"source": "rule_match", "pattern": "conflict"},
                source_chunk_id=chunk.chunk_id,
            ))

        # 检测替代关系：A替代B
        for match in self.REPLACE_PATTERN.finditer(text):
            triples.append(GraphTriple(
                subject=match.group(1),
                subject_type="ingredient",
                predicate="can_replace",
                object=match.group(2),
                object_type="ingredient",
                properties={"source": "rule_match", "pattern": "replace"},
                source_chunk_id=chunk.chunk_id,
            ))

        return triples

    @staticmethod
    def _deduplicate(triples: list[GraphTriple]) -> list[GraphTriple]:
        """去重三元组"""
        seen = set()
        unique = []
        for t in triples:
            key = (t.subject, t.predicate, t.object)
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique
