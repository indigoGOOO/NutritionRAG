"""Content classifier for routing documents to specialized chunkers.

The classifier uses fast rule signals by default and only calls an optional LLM
when the rule result is uncertain or conflicting.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import DOC_TYPE_KEYWORDS
from src.indexing.llm_client import BaseLLMClient
from src.indexing.models import BlockType, DocCategory, TableData, UnifiedDocument

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    primary_category: DocCategory
    confidence: float
    secondary_categories: list[DocCategory] = field(default_factory=list)
    method: str = "rule"
    scores: dict[str, float] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_category": self.primary_category.value,
            "confidence": self.confidence,
            "secondary_categories": [item.value for item in self.secondary_categories],
            "method": self.method,
            "scores": self.scores,
            "signals": self.signals,
            "reason": self.reason,
        }


class ContentClassifier:
    """Rule-first document classifier with optional LLM fallback."""

    CONFIDENCE_THRESHOLD = 0.30
    LLM_FALLBACK_THRESHOLD = 0.45
    TOP_MARGIN_THRESHOLD = 2.0

    TABLE_COLUMN_SIGNATURES = {
        DocCategory.NUTRITION: {
            "热量", "蛋白质", "脂肪", "碳水", "碳水化合物", "钠", "含量", "单位", "NRV",
            "energy", "protein", "fat", "carbohydrate", "sodium",
        },
        DocCategory.RECIPE: {"食材", "用量", "配料", "调料", "步骤", "ingredient", "amount", "step"},
        DocCategory.DAILY: {"日期", "早餐", "午餐", "晚餐", "摄入量", "时间", "date", "breakfast", "lunch", "dinner"},
        DocCategory.PERSONAL: {"姓名", "年龄", "身高", "体重", "过敏源", "过敏史", "BMI", "name", "age", "height", "weight"},
        DocCategory.MEDICAL: {
            "疾病", "症状", "禁忌", "风险", "限制", "指南", "诊疗", "临床", "人群",
            "disease", "symptom", "contraindication", "risk", "guideline", "clinical",
        },
    }

    EXTRA_KEYWORDS = {
        DocCategory.PERSONAL: {
            "个人信息", "用户", "偏好", "过敏", "体重", "身高", "年龄", "BMI",
            "profile", "allergy", "preference",
        },
        DocCategory.DAILY: {
            "每日", "今日", "记录", "早餐", "午餐", "晚餐", "加餐", "日期",
            "daily", "meal log", "breakfast", "lunch", "dinner",
        },
        DocCategory.NUTRITION: {
            "热量", "蛋白质", "脂肪", "碳水", "维生素", "矿物质", "营养成分", "含量",
            "nutrition", "nutrient", "protein", "calorie", "sodium",
        },
        DocCategory.RECIPE: {
            "做法", "步骤", "配料", "食材", "烹饪", "菜谱", "用料", "调料",
            "recipe", "ingredient", "cook", "method",
        },
        DocCategory.MEDICAL: {
            "高血压", "糖尿病", "痛风", "肾病", "妊娠", "孕妇", "儿童", "老人",
            "药物", "禁忌", "诊疗指南", "临床", "疾病", "症状", "风险", "限钠", "控糖",
            "hypertension", "diabetes", "gout", "kidney disease", "pregnancy",
            "medication", "contraindication", "clinical", "guideline",
        },
    }

    STRONG_PATTERNS = {
        DocCategory.PERSONAL: [
            re.compile(r"(姓名|年龄|身高|体重|BMI|过敏史|过敏源).{0,20}(姓名|年龄|身高|体重|BMI|过敏史|过敏源)"),
        ],
        DocCategory.RECIPE: [
            re.compile(r"(步骤\s*\d|第\s*\d+\s*步|\d+[.、]\s*(将|把|加入|倒入|煮|炒))"),
        ],
        DocCategory.MEDICAL: [
            re.compile(r"(诊疗指南|临床指南|疾病|禁忌|高血压|糖尿病|痛风|肾病|药物)"),
        ],
    }

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self.llm = llm_client
        self.keyword_map = self._build_keyword_map()

    def classify(self, document: UnifiedDocument) -> DocCategory:
        """Return the primary category, preserving the legacy API."""
        return self.classify_with_trace(document).primary_category

    def classify_with_trace(self, document: UnifiedDocument) -> ClassificationResult:
        """Classify a document and keep detailed scoring signals."""
        rule_result = self._classify_by_rules(document)
        if self._should_use_llm(rule_result, document):
            llm_result = self._classify_with_llm(document, rule_result)
            if llm_result:
                return llm_result
        return rule_result

    def classify_and_set(self, document: UnifiedDocument) -> UnifiedDocument:
        """Classify and store trace metadata on the document."""
        result = self.classify_with_trace(document)
        document.doc_category = result.primary_category
        document.metadata.extra["classification"] = result.to_dict()
        return document

    def _classify_by_rules(self, document: UnifiedDocument) -> ClassificationResult:
        scores: Counter[DocCategory] = Counter()
        signals: dict[str, Any] = {}

        text = document.text_content
        title_text = self._title_and_path_text(document)

        strong_category, strong_hits = self._score_by_strong_rules(text, title_text)
        if strong_category:
            scores[strong_category] += 10
            signals["strong_rules"] = strong_hits

        keyword_scores, keyword_hits = self._score_by_keywords(f"{title_text}\n{text}")
        scores.update(keyword_scores)
        signals["keyword_hits"] = keyword_hits

        table_scores, table_hits = self._score_by_table_columns(document)
        scores.update(table_scores)
        signals["table_hits"] = table_hits

        structure_scores, structure_hits = self._score_by_structure(document)
        scores.update(structure_scores)
        signals["structure_hits"] = structure_hits

        if not scores:
            return ClassificationResult(
                primary_category=DocCategory.UNKNOWN,
                confidence=0.0,
                method="rule",
                scores={},
                signals=signals,
                reason="no_rule_signal",
            )

        ranked = scores.most_common()
        best_category, best_score = ranked[0]
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score else 0.0
        secondary = [
            category for category, score in ranked[1:3]
            if score > 0 and score / total_score >= 0.2
        ]

        if confidence < self.CONFIDENCE_THRESHOLD:
            best_category = DocCategory.UNKNOWN

        method = "rule_strong" if strong_category == best_category else "rule"
        return ClassificationResult(
            primary_category=best_category,
            confidence=round(confidence, 4),
            secondary_categories=secondary,
            method=method,
            scores={category.value: float(score) for category, score in scores.items()},
            signals=signals,
            reason="rule_scoring",
        )

    def _score_by_keywords(self, text: str) -> tuple[Counter[DocCategory], dict[str, dict[str, int]]]:
        scores: Counter[DocCategory] = Counter()
        hits: dict[str, dict[str, int]] = defaultdict(dict)
        text_lower = text.lower()

        for category, keywords in self.keyword_map.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                count = len(re.findall(re.escape(keyword_lower), text_lower))
                if count:
                    scores[category] += count
                    hits[category.value][keyword] = count

        return scores, dict(hits)

    def _score_by_table_columns(self, document: UnifiedDocument) -> tuple[Counter[DocCategory], dict[str, list[str]]]:
        scores: Counter[DocCategory] = Counter()
        hits: dict[str, list[str]] = defaultdict(list)

        for block in document.blocks:
            if block.block_type != BlockType.TABLE or not isinstance(block.content, TableData):
                continue
            headers_set = {str(header).strip() for header in block.content.headers}
            lower_headers = {header.lower() for header in headers_set}
            for category, signature_cols in self.TABLE_COLUMN_SIGNATURES.items():
                overlap = {
                    header for header in headers_set
                    if header in signature_cols or header.lower() in signature_cols or lower_headers & signature_cols
                }
                if overlap:
                    scores[category] += len(overlap) * 3
                    hits[category.value].extend(sorted(overlap))

        return scores, dict(hits)

    def _score_by_structure(self, document: UnifiedDocument) -> tuple[Counter[DocCategory], dict[str, Any]]:
        scores: Counter[DocCategory] = Counter()
        block_types = [block.block_type for block in document.blocks]
        table_ratio = block_types.count(BlockType.TABLE) / max(len(block_types), 1)
        list_ratio = block_types.count(BlockType.LIST) / max(len(block_types), 1)
        signals = {
            "table_ratio": round(table_ratio, 4),
            "list_ratio": round(list_ratio, 4),
            "step_pattern_count": 0,
        }

        if table_ratio > 0.5:
            scores[DocCategory.NUTRITION] += 2
            scores[DocCategory.DAILY] += 1

        if list_ratio > 0.3:
            scores[DocCategory.RECIPE] += 2

        step_pattern = re.findall(r"(步骤\s*\d|第\s*\d+\s*步|\d+[.、]\s*(将|把|用|加|倒|煮|炒))", document.text_content)
        signals["step_pattern_count"] = len(step_pattern)
        if len(step_pattern) >= 3:
            scores[DocCategory.RECIPE] += 5

        return scores, signals

    def _score_by_strong_rules(
        self,
        text: str,
        title_text: str,
    ) -> tuple[DocCategory | None, dict[str, list[str]]]:
        combined = f"{title_text}\n{text}"
        hits = {}
        for category, patterns in self.STRONG_PATTERNS.items():
            matched = [pattern.pattern for pattern in patterns if pattern.search(combined)]
            if matched:
                hits[category.value] = matched
                return category, hits
        return None, hits

    def _should_use_llm(self, result: ClassificationResult, document: UnifiedDocument) -> bool:
        if self.llm is None:
            return False
        if len(document.text_content.strip()) < 100:
            return False
        score_values = sorted(result.scores.values(), reverse=True)
        top_margin = score_values[0] - score_values[1] if len(score_values) > 1 else score_values[0]
        return (
            result.primary_category == DocCategory.UNKNOWN
            or result.confidence < self.LLM_FALLBACK_THRESHOLD
            or top_margin < self.TOP_MARGIN_THRESHOLD
        )

    def _classify_with_llm(
        self,
        document: UnifiedDocument,
        rule_result: ClassificationResult,
    ) -> ClassificationResult | None:
        prompt = self._build_llm_prompt(document, rule_result)
        try:
            if hasattr(self.llm, "extract_structured"):
                data = self.llm.extract_structured(
                    prompt=prompt,
                    schema={
                        "category": "str",
                        "secondary_categories": [],
                        "confidence": "float",
                        "reason": "str",
                    },
                )
            else:
                raw = self.llm.generate(prompt=prompt)
                data = json.loads(raw)
            category = DocCategory(data.get("category", "unknown"))
            secondary = [
                DocCategory(item) for item in data.get("secondary_categories", [])
                if item in DocCategory._value2member_map_
            ]
            return ClassificationResult(
                primary_category=category,
                confidence=float(data.get("confidence", 0.5) or 0.5),
                secondary_categories=secondary,
                method="llm_fallback",
                scores=rule_result.scores,
                signals={**rule_result.signals, "llm": data},
                reason=data.get("reason", "llm_fallback"),
            )
        except Exception as exc:
            logger.warning("[ContentClassifier] LLM fallback failed: %s", exc)
            return None

    def _build_llm_prompt(self, document: UnifiedDocument, rule_result: ClassificationResult) -> str:
        table_headers = []
        for block in document.blocks:
            if block.block_type == BlockType.TABLE and isinstance(block.content, TableData):
                table_headers.extend(block.content.headers)
        sample = document.text_content[:1500]
        return f"""You are a document classifier for a nutrition assistant.

Categories:
- personal: user profile, allergy, preference, height, weight, personal facts
- daily: daily meal log, meal records, dates, intake records
- nutrition: nutrition facts, food composition, nutrients
- recipe: ingredients, cooking steps, recipes
- medical: disease diet, clinical guideline, contraindication, medication risk, population risk
- unknown: not enough information

Return JSON only:
{{
  "category": "personal|daily|nutrition|recipe|medical|unknown",
  "secondary_categories": [],
  "confidence": 0.0,
  "reason": "short reason"
}}

Title/path: {self._title_and_path_text(document)}
Table headers: {table_headers[:30]}
Rule result: {rule_result.to_dict()}
Text sample:
{sample}
"""

    def _title_and_path_text(self, document: UnifiedDocument) -> str:
        parts = []
        if document.metadata.title:
            parts.append(document.metadata.title)
        if document.metadata.source_path:
            parts.append(Path(document.metadata.source_path).name)
        return " ".join(parts)

    def _build_keyword_map(self) -> dict[DocCategory, set[str]]:
        keyword_map: dict[DocCategory, set[str]] = {category: set() for category in DocCategory}
        for category_str, keywords in DOC_TYPE_KEYWORDS.items():
            if category_str in DocCategory._value2member_map_:
                keyword_map[DocCategory(category_str)].update(str(item) for item in keywords)
        for category, keywords in self.EXTRA_KEYWORDS.items():
            keyword_map[category].update(keywords)
        return keyword_map

