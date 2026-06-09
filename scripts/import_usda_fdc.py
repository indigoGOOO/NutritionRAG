"""Import USDA FoodData Central records into PostgreSQL.

Input: local .json/.jsonl/.csv records with FDC-like fields.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.normalizer import merge_batches, normalize_usda_food
from src.data_ingestion.readers import read_records
from src.data_ingestion.schemas import SourceMetadata
from src.data_ingestion.writers import PGIngestionWriter
from src.storage.pg_client import PostgreSQLClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Import USDA FDC nutrition data")
    parser.add_argument("input", type=Path, help="Path to .json/.jsonl/.csv records")
    parser.add_argument("--source-url", default="https://fdc.nal.usda.gov/")
    parser.add_argument("--init-tables", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records = read_records(args.input)
    if args.limit:
        records = records[: args.limit]

    source = SourceMetadata(
        source="USDA FoodData Central",
        source_type="official_database",
        source_url=args.source_url,
        language="en",
        confidence=0.95,
    )
    batch = merge_batches([normalize_usda_food(record, source) for record in records])

    pg = PostgreSQLClient()
    if args.init_tables:
        pg.init_tables()
    stats = PGIngestionWriter(pg).write_batch(batch)
    print({
        "input_records": len(records),
        "entities": len(batch.entities),
        "nutrient_profiles": len(batch.nutrient_profiles),
        **stats,
    })


if __name__ == "__main__":
    main()

