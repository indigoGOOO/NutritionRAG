"""Writers that persist canonical import batches into the existing stores."""

from __future__ import annotations

from src.data_ingestion.schemas import FoodEntity, ImportBatch, NutrientProfile, RelationFact
from src.storage.pg_client import PostgreSQLClient


def source_metadata_dict(item) -> dict:
    if item.source is None:
        return {}
    return item.source.to_dict()


def nutrient_profile_to_kv(profile: NutrientProfile) -> dict:
    metadata = source_metadata_dict(profile)
    return {
        "entity_id": profile.entity_id,
        "entity_type": profile.entity_type,
        "attribute": "nutrients",
        "value": {
            **profile.nutrients,
            "basis": profile.serving_basis,
        },
        "confidence": metadata.get("confidence"),
        "metadata": metadata,
    }


def entity_type_to_kv(entity: FoodEntity) -> dict:
    metadata = source_metadata_dict(entity)
    return {
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "attribute": "_entity_type",
        "value": {
            "type": entity.entity_type,
            "name": entity.name,
            "aliases": entity.aliases,
        },
        "confidence": metadata.get("confidence"),
        "metadata": metadata,
    }


def relation_to_triple(relation: RelationFact) -> dict:
    metadata = source_metadata_dict(relation)
    metadata.update({
        "subject_type": relation.subject_type,
        "object_type": relation.object_type,
    })
    return {
        "subject": relation.subject,
        "predicate": relation.predicate,
        "object": relation.object,
        "confidence": relation.confidence,
        "metadata": metadata,
    }


class PGIngestionWriter:
    """Persist canonical batches into PostgreSQL."""

    def __init__(self, pg: PostgreSQLClient):
        self.pg = pg

    def write_batch(self, batch: ImportBatch) -> dict:
        kv_rows = [entity_type_to_kv(entity) for entity in batch.entities]
        kv_rows.extend(nutrient_profile_to_kv(profile) for profile in batch.nutrient_profiles)

        triple_rows = [relation_to_triple(relation) for relation in batch.relations]

        alias_rows = []
        for entity in batch.entities:
            metadata = source_metadata_dict(entity)
            aliases = [entity.name, *entity.aliases]
            for alias in _dedupe(aliases):
                alias_rows.append({
                    "canonical_entity_id": entity.entity_id,
                    "alias": alias,
                    "alias_type": "name",
                    "language": metadata.get("language", ""),
                    "source": metadata.get("source", ""),
                    "confidence": metadata.get("confidence", 0.8),
                })

        kv_ids = self.pg.insert_kv_pairs_batch(kv_rows) if kv_rows else []
        triple_ids = self.pg.insert_triples_batch(triple_rows) if triple_rows else []
        alias_ids = self.pg.insert_entity_aliases_batch(alias_rows) if alias_rows else []

        return {
            "kv_count": len(kv_ids),
            "triple_count": len(triple_ids),
            "alias_count": len(alias_ids),
        }


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result

