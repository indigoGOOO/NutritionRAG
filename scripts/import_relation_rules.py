"""Import curated relation rules from CSV into PostgreSQL triples.

CSV columns:
subject,predicate,object,subject_type,object_type,confidence,source,source_url,language
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.readers import read_csv_records
from src.data_ingestion.schemas import ImportBatch, RelationFact, SourceMetadata
from src.data_ingestion.writers import PGIngestionWriter
from src.storage.pg_client import PostgreSQLClient


def _to_relation(row: dict) -> RelationFact | None:
    subject = (row.get("subject") or "").strip()
    predicate = (row.get("predicate") or "").strip()
    obj = (row.get("object") or "").strip()
    if not subject or not predicate or not obj:
        return None
    confidence = float(row.get("confidence") or 0.85)
    source = SourceMetadata(
        source=row.get("source") or "curated_rules",
        source_type=row.get("source_type") or "curated_rule",
        source_url=row.get("source_url") or "",
        language=row.get("language") or "zh",
        confidence=confidence,
        extra={"rule_type": row.get("rule_type") or ""},
    )
    return RelationFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        subject_type=row.get("subject_type") or "unknown",
        object_type=row.get("object_type") or "unknown",
        confidence=confidence,
        source=source,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import curated relation rules")
    parser.add_argument("input", type=Path, help="Path to relation rules CSV")
    parser.add_argument("--init-tables", action="store_true")
    args = parser.parse_args()

    relations = []
    for row in read_csv_records(args.input):
        relation = _to_relation(row)
        if relation:
            relations.append(relation)

    pg = PostgreSQLClient()
    if args.init_tables:
        pg.init_tables()
    stats = PGIngestionWriter(pg).write_batch(ImportBatch(relations=relations))
    print({"relations": len(relations), **stats})


if __name__ == "__main__":
    main()

