"""记忆层数据模型与抽象基类

三种记忆类型：
- conversation: 对话历史（谁在什么时候说了什么）
- profile: 用户画像（过敏源、偏好、饮食限制）
- knowledge: 长期知识（有价值的问答对，可复用）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class MemoryItem:
    """通用记忆条目"""
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    score: float = 0.0
    source: str = ""  # conversation / profile / knowledge


@dataclass
class ConversationTurn:
    """单轮对话"""
    role: str  # user | assistant
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    preferences: dict[str, str] = field(default_factory=dict)  # 偏好
    allergies: list[str] = field(default_factory=list)  # 过敏源
    dietary_restrictions: list[str] = field(default_factory=list)  # 饮食限制
    health_goals: list[str] = field(default_factory=list)  # 健康目标
    favorite_ingredients: list[str] = field(default_factory=list)  # 偏好食材
    disliked_ingredients: list[str] = field(default_factory=list)  # 不喜欢的食材
    notes: str = ""


@dataclass
class KnowledgeEntry:
    """长期知识条目（有价值的问答对）"""
    id: str
    question: str
    answer: str
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    hit_count: int = 0
    rating: float = 0.0  # 用户反馈评分
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)


class BaseMemory(ABC):
    """记忆存储基类"""

    @abstractmethod
    def add(self, item: MemoryItem) -> str:
        """写入一条记忆，返回记忆ID"""
        ...

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """检索相关记忆"""
        ...

    @abstractmethod
    def remove(self, item_id: str) -> bool:
        """删除指定记忆"""
        ...
