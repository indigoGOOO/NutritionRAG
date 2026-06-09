"""Import food-compound rows, e.g. FooDB exports, into PG triples.

CSV columns:
food,compound,food_type,compound_type,confidence,source,source_url,language
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.normalizer import canonical_entity_id
from src.data_ingestion.readers import read_csv_records
from src.data_ingestion.schemas import FoodEntity, ImportBatch, RelationFact, SourceMetadata
from src.data_ingestion.writers import PGIngestionWriter
from src.storage.pg_client import PostgreSQLClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Import food-compound relations")
    parser.add_argument("input", type=Path, help="Path to food-compound CSV")
    parser.add_argument("--init-tables", action="store_true")
    args = parser.parse_args()

    entities = []
    relations = []
    seen_entities = set()
    for row in read_csv_records(args.input):
        food = (row.get("food") or "").strip()
        compound = (row.get("compound") or "").strip()
        if not food or not compound:
            continue
        confidence = float(row.get("confidence") or 0.8)
        source = SourceMetadata(
            source=row.get("source") or "FooDB",
            source_type=row.get("source_type") or "food_compound_database",
            source_url=row.get("source_url") or "https://foodb.ca/",
            language=row.get("language") or "en",
            confidence=confidence,
        )
        food_id = canonical_entity_id(food)
        compound_id = canonical_entity_id(compound)
        if food_id not in seen_entities:
            entities.append(FoodEntity(food_id, food, row.get("food_type") or "food", [food], source))
            seen_entities.add(food_id)
        if compound_id not in seen_entities:
            entities.append(FoodEntity(
                compound_id,
                compound,
                row.get("compound_type") or "compound",
                [compound],
                source,
            ))
            seen_entities.add(compound_id)
        relations.append(RelationFact(
            subject=food_id,
            predicate="contains",
            object=compound_id,
            subject_type=row.get("food_type") or "food",
            object_type=row.get("compound_type") or "compound",
            confidence=confidence,
            source=source,
        ))

    pg = PostgreSQLClient()
    if args.init_tables:
        pg.init_tables()
    stats = PGIngestionWriter(pg).write_batch(ImportBatch(entities=entities, relations=relations))
    print({"entities": len(entities), "relations": len(relations), **stats})


if __name__ == "__main__":
    main()

