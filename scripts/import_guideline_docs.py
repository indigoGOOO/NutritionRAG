"""Batch import guideline/doc files through the existing indexing pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.chunk_router import ChunkPurpose
from src.indexing.index_pipeline import IndexingPipeline
from src.storage.storage_manager import StorageManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Import guideline documents")
    parser.add_argument("input", type=Path, help="File or directory")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--init-stores", action="store_true")
    args = parser.parse_args()

    pipeline = IndexingPipeline()
    storage = StorageManager()
    if args.init_stores:
        storage.init_all()

    if args.input.is_dir():
        results = pipeline.run_batch(
            args.input,
            recursive=args.recursive,
            chunk_purpose=ChunkPurpose.RETRIEVAL,
        )
    else:
        results = [pipeline.run(args.input, chunk_purpose=ChunkPurpose.RETRIEVAL)]

    stored = []
    for result in results:
        if result.errors:
            print({"doc_id": result.doc_id, "errors": result.errors})
        stored.append(storage.store_pipeline_result(result))

    print({
        "documents": len(results),
        "stored": stored,
    })


if __name__ == "__main__":
    main()

