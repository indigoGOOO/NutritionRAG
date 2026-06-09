"""Lightweight classifier for user-owned content types."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from src.user_content.models import UserContentClassification, UserContentType


SAVE_VERBS = ("保存", "存一下", "记一下", "帮我记", "记录一下", "save", "remember")

TYPE_KEYWORDS = {
    UserContentType.RECIPE: {
        "菜谱", "食谱", "做法", "步骤", "食材", "配料", "调料", "烹饪", "recipe", "ingredient",
    },
    UserContentType.MEAL_PLAN: {
        "饮食计划", "菜单", "一周菜单", "早餐", "午餐", "晚餐", "加餐", "meal plan", "menu",
    },
    UserContentType.WORKOUT_PLAN: {
        "训练计划", "健身计划", "力量训练", "有氧", "深蹲", "卧推", "跑步", "组", "次数",
        "workout", "training", "exercise",
    },
    UserContentType.FOOD_LOG: {
        "饮食记录", "今天吃了", "早餐吃", "午餐吃", "晚餐吃", "摄入", "food log", "ate",
    },
    UserContentType.BODY_METRICS: {
        "体重", "身高", "BMI", "体脂", "腰围", "血压", "心率", "body weight", "body fat",
    },
    UserContentType.LAB_REPORT: {
        "体检", "化验", "报告", "血糖", "血脂", "尿酸", "肌酐", "胆固醇", "甘油三酯",
        "lab report", "blood test",
    },
}

STRONG_PATTERNS = {
    UserContentType.RECIPE: re.compile(r"(步骤\s*\d|第\s*\d+\s*步|食材[:：]|配料[:：])"),
    UserContentType.MEAL_PLAN: re.compile(r"(周[一二三四五六日天]|星期[一二三四五六日天]|早餐|午餐|晚餐).{0,20}(早餐|午餐|晚餐)"),
    UserContentType.WORKOUT_PLAN: re.compile(r"(训练|健身|workout).{0,30}(组|次数|分钟|有氧|力量)"),
    UserContentType.FOOD_LOG: re.compile(r"(今天|昨日|昨天).{0,10}(吃了|早餐|午餐|晚餐)"),
    UserContentType.BODY_METRICS: re.compile(r"(体重|身高|BMI|体脂|腰围|血压)[:：]?\s*\d+"),
    UserContentType.LAB_REPORT: re.compile(r"(体检报告|化验单|血糖|血脂|尿酸|肌酐|胆固醇)"),
}


class UserContentClassifier:
    """Classify text into one of the supported user-owned content types."""

    def classify(self, text: str, explicit_type: str | None = None) -> UserContentClassification:
        if explicit_type:
            content_type = _parse_content_type(explicit_type)
            if content_type:
                return UserContentClassification(
                    content_type=content_type,
                    confidence=1.0,
                    reason="explicit_content_type",
                    signals={"explicit_type": explicit_type},
                )

        lowered = text.lower()
        scores: Counter[UserContentType] = Counter()
        signals = defaultdict(dict)

        for content_type, pattern in STRONG_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                scores[content_type] += 5
                signals[content_type.value]["strong_pattern"] = len(matches)

        for content_type, keywords in TYPE_KEYWORDS.items():
            for keyword in keywords:
                count = lowered.count(keyword.lower())
                if count:
                    scores[content_type] += count
                    signals[content_type.value][keyword] = count

        if not scores:
            return UserContentClassification(None, 0.0, "no_content_type_signal", {})

        best, best_score = scores.most_common(1)[0]
        total = sum(scores.values())
        confidence = best_score / total if total else 0.0
        return UserContentClassification(
            content_type=best,
            confidence=round(confidence, 4),
            reason="rule_scoring",
            signals=dict(signals),
        )

    def is_explicit_save_request(self, text: str) -> bool:
        lowered = text.lower()
        return any(verb in text or verb in lowered for verb in SAVE_VERBS)


def _parse_content_type(value: str) -> UserContentType | None:
    try:
        return UserContentType(value)
    except ValueError:
        return None

