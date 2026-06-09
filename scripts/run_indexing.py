"""索引管线入口脚本

用法:
    python scripts/run_indexing.py data/raw/
    python scripts/run_indexing.py data/raw/营养成分表.pdf
    python scripts/run_indexing.py --text "用户喜欢清淡饮食，过敏源是海鲜"
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.index_pipeline import IndexingPipeline
from src.storage.storage_manager import StorageManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="营养RAG索引管线")
    parser.add_argument("input", help="输入文件或目录路径")
    parser.add_argument("--recursive", "-r", action="store_true", help="递归处理子目录")
    parser.add_argument("--text", "-t", help="直接处理文本字符串")
    parser.add_argument("--source", "-s", default="cli", help="文本来源名称")
    parser.add_argument("--store", action="store_true", help="将索引结果写入PostgreSQL/Milvus/Neo4j")
    parser.add_argument("--init-store", action="store_true", help="写入前初始化数据库表、集合和图约束")
    args = parser.parse_args()

    pipeline = IndexingPipeline()
    storage = None
    if args.store:
        storage = StorageManager()
        if args.init_store:
            storage.init_all()

    input_path = Path(args.input)

    if args.text:
        logger.info(f"处理文本输入: {args.text[:50]}...")
        result = pipeline.run_text(args.text, source_name=args.source)
    elif input_path.is_file():
        logger.info(f"处理文件: {input_path}")
        result = pipeline.run(input_path)
    elif input_path.is_dir():
        logger.info(f"批量处理目录: {input_path}")
        results = pipeline.run_batch(input_path, recursive=args.recursive)
        store_summaries = []
        if storage:
            for r in results:
                if r.chunks or r.kv_pairs or r.triples:
                    store_summaries.append(storage.store_pipeline_result(r))
        print(f"\n{'='*50}")
        print(f"批量处理完成: {len(results)} 个文档")
        for r in results:
            status = "OK" if not r.errors else f"ERROR ({len(r.errors)})"
            print(f"  [{status}] {r.doc_id[:8]} - {r.stats.get('chunk_count', 0)} chunks")
        if store_summaries:
            print(f"已入库文档: {len(store_summaries)}")
        if storage:
            storage.close_all()
        return
    else:
        logger.error(f"路径不存在: {input_path}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"文档ID: {result.doc_id}")
    print(f"Chunks: {result.stats.get('chunk_count', 0)}")
    print(f"KV Pairs: {result.stats.get('kv_count', 0)}")
    print(f"Triples: {result.stats.get('triple_count', 0)}")
    print(f"耗时: {result.stats.get('elapsed_seconds', 0)}s")

    if result.errors:
        print(f"\n错误 ({len(result.errors)}):")
        for err in result.errors:
            print(f"  - {err}")
    elif storage:
        stored = storage.store_pipeline_result(result)
        print("\n入库完成:")
        print(f"  Chunks: {stored['chunks']}")
        print(f"  KV Pairs: {stored['kv_pairs']}")
        print(f"  Triples: {stored['triples']}")

    if storage:
        storage.close_all()


if __name__ == "__main__":
    main()
