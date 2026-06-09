"""用户画像记忆 - PostgreSQL 主存储

设计原则：
- PostgreSQL 的 user_profiles 是 source of truth，保证画像始终可读可改
- 不再把用户画像混入知识库 kv_pairs
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.memory.base import BaseMemory, MemoryItem, UserProfile
from src.storage.pg_client import PostgreSQLClient

logger = logging.getLogger(__name__)


class ProfileMemory(BaseMemory):
    """用户画像记忆"""

    def __init__(self, pg: PostgreSQLClient, neo4j: Optional["Neo4jClient"] = None):
        self.pg = pg
        self.neo4j = neo4j
        self._init_table()

    def _init_table(self):
        with self.pg.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id VARCHAR(255) PRIMARY KEY,
                    allergies JSONB DEFAULT '[]',
                    dietary_restrictions JSONB DEFAULT '[]',
                    health_goals JSONB DEFAULT '[]',
                    favorite_ingredients JSONB DEFAULT '[]',
                    disliked_ingredients JSONB DEFAULT '[]',
                    preferences JSONB DEFAULT '{}',
                    notes TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        self.pg.conn.commit()

    # ==================== PostgreSQL source of truth ====================

    def ensure_profile(self, user_id: str):
        with self.pg.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO user_profiles (user_id)
                   VALUES (%s)
                   ON CONFLICT (user_id) DO NOTHING""",
                (user_id,),
            )
        self.pg.conn.commit()

    def set_preference(self, user_id: str, key: str, value: Any):
        """设置用户偏好/属性，优先写入专用字段，其余写入preferences JSON。"""
        self.ensure_profile(user_id)
        field_map = {
            "allergies": "allergies",
            "dietary_restrictions": "dietary_restrictions",
            "health_goals": "health_goals",
            "favorite_ingredients": "favorite_ingredients",
            "disliked_ingredients": "disliked_ingredients",
            "notes": "notes",
        }
        if key in field_map and key != "notes":
            self._set_jsonb_list(user_id, field_map[key], self._coerce_list(value))
        elif key == "notes":
            self._set_notes(user_id, str(value))
        else:
            prefs = self.get_preferences(user_id)
            prefs[key] = value
            with self.pg.conn.cursor() as cur:
                cur.execute(
                    """UPDATE user_profiles
                       SET preferences = %s, updated_at = CURRENT_TIMESTAMP
                       WHERE user_id = %s""",
                    (json.dumps(prefs, ensure_ascii=False), user_id),
                )
            self.pg.conn.commit()
        logger.info(f"[Profile] 设置画像: {user_id} / {key}={value}")

    def get_preferences(self, user_id: str) -> dict[str, Any]:
        row = self._get_profile_row(user_id)
        if not row:
            return {}
        profile = self._row_to_profile_dict(row)
        prefs = profile.get("preferences", {}) or {}
        for key in (
            "allergies",
            "dietary_restrictions",
            "health_goals",
            "favorite_ingredients",
            "disliked_ingredients",
        ):
            prefs[key] = profile.get(key, [])
        if profile.get("notes"):
            prefs["notes"] = profile["notes"]
        return prefs

    def add_allergy(self, user_id: str, ingredient_name: str):
        """记录过敏：先写PG，再尽力同步Neo4j。"""
        self._append_jsonb_list(user_id, "allergies", [ingredient_name])
        self._safe_sync_allergy_to_neo4j(user_id, ingredient_name)
        logger.info(f"[Profile] 记录过敏: {user_id} → {ingredient_name}")

    def add_dietary_restriction(
        self,
        user_id: str,
        restriction: str,
        ingredient: str | None = None,
    ):
        """记录饮食限制：先写PG，再尽力同步Neo4j。"""
        self._append_jsonb_list(user_id, "dietary_restrictions", [restriction])
        if ingredient:
            self._safe_sync_restriction_to_neo4j(user_id, restriction, ingredient)
        logger.info(f"[Profile] 记录饮食限制: {user_id} / {restriction}")

    def add_health_goal(self, user_id: str, goal: str):
        self._append_jsonb_list(user_id, "health_goals", [goal])
        self._safe_sync_simple_relation(user_id, "HAS_GOAL", goal, "Goal")

    def add_favorite_ingredient(self, user_id: str, ingredient: str):
        self._append_jsonb_list(user_id, "favorite_ingredients", [ingredient])
        self._safe_sync_simple_relation(user_id, "LIKES", ingredient, "Food")

    def add_disliked_ingredient(self, user_id: str, ingredient: str):
        self._append_jsonb_list(user_id, "disliked_ingredients", [ingredient])
        self._safe_sync_simple_relation(user_id, "DISLIKES", ingredient, "Food")

    def get_allergies(self, user_id: str) -> list[str]:
        return self._get_jsonb_list(user_id, "allergies")

    def get_restrictions(self, user_id: str) -> list[dict]:
        return [
            {"ingredient": "", "restriction": item}
            for item in self._get_jsonb_list(user_id, "dietary_restrictions")
        ]

    def check_ingredient_safety(self, user_id: str, ingredient: str) -> dict:
        allergies = self.get_allergies(user_id)
        if ingredient in allergies:
            return {"safe": False, "reason": f"过敏源: {ingredient}", "type": "allergy"}

        restrictions = self.get_restrictions(user_id)
        for r in restrictions:
            if r["ingredient"] == ingredient:
                return {
                    "safe": False,
                    "reason": f"饮食限制({r['restriction']}): {ingredient}",
                    "type": "restriction",
                }
        return {"safe": True, "reason": "", "type": None}

    def get_full_profile(self, user_id: str) -> UserProfile:
        row = self._get_profile_row(user_id)
        if not row:
            return UserProfile(user_id=user_id)
        data = self._row_to_profile_dict(row)
        return UserProfile(
            user_id=user_id,
            preferences=data.get("preferences", {}),
            allergies=data.get("allergies", []),
            dietary_restrictions=data.get("dietary_restrictions", []),
            health_goals=data.get("health_goals", []),
            favorite_ingredients=data.get("favorite_ingredients", []),
            disliked_ingredients=data.get("disliked_ingredients", []),
            notes=data.get("notes", ""),
        )

    def profile_to_text(self, user_id: str) -> str:
        profile = self.get_full_profile(user_id)

        parts = []
        if profile.allergies:
            parts.append(f"过敏源：{'、'.join(profile.allergies)}")
        if profile.dietary_restrictions:
            parts.append(f"饮食限制：{'、'.join(profile.dietary_restrictions)}")
        if profile.health_goals:
            parts.append(f"健康目标：{'、'.join(profile.health_goals)}")
        if profile.favorite_ingredients:
            parts.append(f"偏好食材：{'、'.join(profile.favorite_ingredients)}")
        if profile.disliked_ingredients:
            parts.append(f"不喜欢的食材：{'、'.join(profile.disliked_ingredients)}")
        if profile.notes:
            parts.append(f"备注：{profile.notes}")

        return "；".join(parts) if parts else "暂无用户画像信息"

    # ==================== Neo4j best-effort sync ====================

    def _ensure_user_node(self, user_id: str):
        if self.neo4j is None:
            return
        existing = self.neo4j.query_entity(f"user_{user_id}")
        if not existing:
            self.neo4j.create_entity(
                entity_id=f"user_{user_id}",
                entity_type="User",
                name=user_id,
            )

    def _safe_sync_allergy_to_neo4j(self, user_id: str, ingredient_name: str):
        if self.neo4j is None:
            return
        try:
            self._ensure_user_node(user_id)
            food_id = f"food_{ingredient_name}"
            if not self.neo4j.query_entity(food_id):
                self.neo4j.create_entity(food_id, "Food", ingredient_name)
            self.neo4j.create_relationship(
                subject_id=f"user_{user_id}",
                predicate="ALLERGIC_TO",
                object_id=food_id,
                properties={"source": "profile_memory"},
            )
        except Exception as e:
            logger.warning(f"[Profile] Neo4j同步过敏失败: {e}")

    def _safe_sync_restriction_to_neo4j(self, user_id: str, restriction: str, ingredient: str):
        if self.neo4j is None:
            return
        try:
            self._ensure_user_node(user_id)
            food_id = f"food_{ingredient}"
            if not self.neo4j.query_entity(food_id):
                self.neo4j.create_entity(food_id, "Food", ingredient)
            self.neo4j.create_relationship(
                subject_id=f"user_{user_id}",
                predicate="RESTRICTED_TO",
                object_id=food_id,
                properties={"restriction": restriction, "source": "profile_memory"},
            )
        except Exception as e:
            logger.warning(f"[Profile] Neo4j同步饮食限制失败: {e}")

    def _safe_sync_simple_relation(self, user_id: str, predicate: str, value: str, entity_type: str):
        if self.neo4j is None:
            return
        try:
            self._ensure_user_node(user_id)
            entity_id = f"{entity_type.lower()}_{value}"
            if not self.neo4j.query_entity(entity_id):
                self.neo4j.create_entity(entity_id, entity_type, value)
            self.neo4j.create_relationship(
                subject_id=f"user_{user_id}",
                predicate=predicate,
                object_id=entity_id,
                properties={"source": "profile_memory"},
            )
        except Exception as e:
            logger.warning(f"[Profile] Neo4j同步关系失败 {predicate}/{value}: {e}")

    # ==================== BaseMemory ====================

    def add(self, item: MemoryItem) -> str:
        self.set_preference(
            item.metadata.get("user_id", "default"),
            item.metadata.get("key", "notes"),
            item.content,
        )
        return item.id

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        return []

    def remove(self, item_id: str) -> bool:
        return False

    # ==================== helpers ====================

    def _get_profile_row(self, user_id: str) -> tuple | None:
        with self.pg.conn.cursor() as cur:
            cur.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
            return cur.fetchone()

    @staticmethod
    def _row_to_profile_dict(row: tuple) -> dict[str, Any]:
        names = [
            "user_id",
            "allergies",
            "dietary_restrictions",
            "health_goals",
            "favorite_ingredients",
            "disliked_ingredients",
            "preferences",
            "notes",
            "created_at",
            "updated_at",
        ]
        data = dict(zip(names, row))
        for field in (
            "allergies",
            "dietary_restrictions",
            "health_goals",
            "favorite_ingredients",
            "disliked_ingredients",
            "preferences",
        ):
            if isinstance(data.get(field), str):
                data[field] = json.loads(data[field])
            elif data.get(field) is None:
                data[field] = {} if field == "preferences" else []
        return data

    def _get_jsonb_list(self, user_id: str, field: str) -> list[str]:
        row = self._get_profile_row(user_id)
        if not row:
            return []
        data = self._row_to_profile_dict(row)
        return [str(v) for v in data.get(field, [])]

    def _set_jsonb_list(self, user_id: str, field: str, values: list[str]):
        self.ensure_profile(user_id)
        with self.pg.conn.cursor() as cur:
            cur.execute(
                f"""UPDATE user_profiles
                    SET {field} = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s""",
                (json.dumps(values, ensure_ascii=False), user_id),
            )
        self.pg.conn.commit()

    def _append_jsonb_list(self, user_id: str, field: str, values: list[str]):
        current = self._get_jsonb_list(user_id, field)
        merged = []
        for value in [*current, *values]:
            value = str(value).strip()
            if value and value not in merged:
                merged.append(value)
        self._set_jsonb_list(user_id, field, merged)

    def _set_notes(self, user_id: str, notes: str):
        self.ensure_profile(user_id)
        with self.pg.conn.cursor() as cur:
            cur.execute(
                """UPDATE user_profiles
                   SET notes = %s, updated_at = CURRENT_TIMESTAMP
                   WHERE user_id = %s""",
                (notes, user_id),
            )
        self.pg.conn.commit()

    @staticmethod
    def _coerce_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return [str(value)] if value is not None else []
