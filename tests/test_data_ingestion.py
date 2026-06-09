import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.normalizer import (
    canonical_entity_id,
    normalize_openfoodfacts_product,
    normalize_usda_food,
)
from src.data_ingestion.schemas import FoodEntity, ImportBatch, NutrientProfile, SourceMetadata
from src.data_ingestion.writers import PGIngestionWriter


def test_canonical_entity_id_keeps_chinese_and_normalizes_spaces():
    assert canonical_entity_id("Chicken Breast", "usda") == "usda_chicken_breast"
    assert canonical_entity_id("鸡胸肉", "") == "鸡胸肉"


def test_normalize_usda_food_maps_nutrients():
    source = SourceMetadata("USDA FoodData Central", "official_database")
    record = {
        "fdcId": 123,
        "description": "Chicken breast",
        "foodNutrients": [
            {"nutrientName": "Protein", "amount": 31},
            {"nutrientName": "Sodium, Na", "amount": 74},
        ],
    }

    batch = normalize_usda_food(record, source)

    assert batch.entities[0].entity_id == "usda_chicken_breast"
    assert batch.nutrient_profiles[0].nutrients == {
        "protein_g": 31,
        "sodium_mg": 74,
    }
    assert batch.nutrient_profiles[0].source.to_dict()["fdc_id"] == "123"


def test_normalize_openfoodfacts_product_maps_label_fields():
    source = SourceMetadata("Open Food Facts", "crowdsourced_database")
    record = {
        "code": "12345",
        "product_name": "Protein Bar",
        "brands": "Demo",
        "ingredients_text": "oats, milk",
        "nutriments": {"proteins_100g": 20, "fat_100g": 7},
    }

    batch = normalize_openfoodfacts_product(record, source)

    assert batch.entities[0].entity_id == "off_12345"
    assert "Demo Protein Bar" in batch.entities[0].aliases
    assert batch.nutrient_profiles[0].nutrients["protein_g"] == 20
    assert batch.nutrient_profiles[0].nutrients["fat_g"] == 7


class FakePG:
    def __init__(self):
        self.kv_rows = []
        self.triple_rows = []
        self.alias_rows = []

    def insert_kv_pairs_batch(self, rows):
        self.kv_rows = rows
        return list(range(1, len(rows) + 1))

    def insert_triples_batch(self, rows):
        self.triple_rows = rows
        return list(range(1, len(rows) + 1))

    def insert_entity_aliases_batch(self, rows):
        self.alias_rows = rows
        return list(range(1, len(rows) + 1))


def test_pg_ingestion_writer_writes_kv_and_aliases():
    source = SourceMetadata("test", "fixture", confidence=0.9, language="zh")
    batch = ImportBatch(
        entities=[
            FoodEntity(
                entity_id="chicken_breast",
                name="鸡胸肉",
                entity_type="food",
                aliases=["chicken breast"],
                source=source,
            )
        ],
        nutrient_profiles=[
            NutrientProfile(
                entity_id="chicken_breast",
                nutrients={"protein_g": 31},
                source=source,
            )
        ],
    )
    pg = FakePG()

    stats = PGIngestionWriter(pg).write_batch(batch)

    assert stats == {"kv_count": 2, "triple_count": 0, "alias_count": 2}
    assert pg.kv_rows[0]["attribute"] == "_entity_type"
    assert pg.kv_rows[1]["attribute"] == "nutrients"
    assert pg.kv_rows[1]["metadata"]["source"] == "test"
    assert pg.alias_rows == [
        {
            "canonical_entity_id": "chicken_breast",
            "alias": "鸡胸肉",
            "alias_type": "name",
            "language": "zh",
            "source": "test",
            "confidence": 0.9,
        },
        {
            "canonical_entity_id": "chicken_breast",
            "alias": "chicken breast",
            "alias_type": "name",
            "language": "zh",
            "source": "test",
            "confidence": 0.9,
        },
    ]

