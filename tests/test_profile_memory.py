import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.profile_memory import ProfileMemory


class FailingNeo4j:
    def query_entity(self, entity_id):
        raise RuntimeError("neo4j unavailable")


def make_profile_without_db():
    profile = ProfileMemory.__new__(ProfileMemory)
    profile.neo4j = FailingNeo4j()
    profile.pg = None
    profile.appended = []

    def append_jsonb_list(user_id, field, values):
        profile.appended.append((user_id, field, values))

    profile._append_jsonb_list = append_jsonb_list
    return profile


def test_add_allergy_writes_pg_before_best_effort_neo4j_sync():
    profile = make_profile_without_db()

    profile.add_allergy("u1", "peanut")

    assert profile.appended == [("u1", "allergies", ["peanut"])]


def test_dedicated_profile_fields_write_pg_even_when_neo4j_fails():
    profile = make_profile_without_db()

    profile.add_favorite_ingredient("u1", "oat")
    profile.add_disliked_ingredient("u1", "pepper")
    profile.add_health_goal("u1", "low_sugar")

    assert profile.appended == [
        ("u1", "favorite_ingredients", ["oat"]),
        ("u1", "disliked_ingredients", ["pepper"]),
        ("u1", "health_goals", ["low_sugar"]),
    ]


def test_row_to_profile_dict_decodes_user_profiles_json_fields():
    row = (
        "u1",
        '["peanut"]',
        '["low_salt"]',
        '["weight_control"]',
        '["oat"]',
        '["pepper"]',
        '{"cuisine": "light"}',
        "note",
        None,
        None,
    )

    data = ProfileMemory._row_to_profile_dict(row)

    assert data["allergies"] == ["peanut"]
    assert data["dietary_restrictions"] == ["low_salt"]
    assert data["health_goals"] == ["weight_control"]
    assert data["favorite_ingredients"] == ["oat"]
    assert data["disliked_ingredients"] == ["pepper"]
    assert data["preferences"] == {"cuisine": "light"}
    assert data["notes"] == "note"


def test_coerce_list_accepts_legacy_comma_strings():
    assert ProfileMemory._coerce_list("a,b, c") == ["a", "b", "c"]
    assert ProfileMemory._coerce_list(["a", ""]) == ["a"]
