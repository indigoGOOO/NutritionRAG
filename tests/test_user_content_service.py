import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.models import ContentChunk, DocCategory, GraphTriple, KVPair, PipelineResult
from src.user_content.classifier import UserContentClassifier
from src.user_content.service import UserContentService


def test_user_content_classifier_detects_supported_types():
    classifier = UserContentClassifier()

    assert classifier.classify("保存这个菜谱：食材 鸡蛋，步骤1 打散。").content_type.value == "recipe"
    assert classifier.classify("帮我记一下训练计划：深蹲 4组，每组10次").content_type.value == "workout_plan"
    assert classifier.classify("记录一下体重 70kg，体脂 18%").content_type.value == "body_metrics"
    assert classifier.classify("保存体检报告：尿酸 500，血糖 6.1").content_type.value == "lab_report"


class FakePipeline:
    def run_text(self, text, source_name):
        return PipelineResult(
            doc_id="old",
            chunks=[
                ContentChunk(
                    content=text,
                    chunk_type="text",
                    doc_category=DocCategory.UNKNOWN,
                    source_doc_id="old",
                    token_count=80,
                )
            ],
            kv_pairs=[
                KVPair(
                    key="recipe",
                    entity_type="",
                    value={"attributes": []},
                    source_doc_id="old",
                )
            ],
            triples=[
                GraphTriple(
                    subject="recipe",
                    predicate="contains",
                    object="egg",
                    properties={},
                    source_chunk_id="missing",
                )
            ],
            embeddings=[
                {"chunk_id": "", "dense_vector": [0.1, 0.2], "sparse_vector": {}}
            ],
        )


class FakeStorage:
    def __init__(self):
        self.result = None

    def store_pipeline_result(self, result):
        self.result = result
        return {"chunks": len(result.chunks), "kv_pairs": len(result.kv_pairs), "triples": len(result.triples)}


def test_user_content_service_saves_with_user_metadata():
    storage = FakeStorage()
    service = UserContentService(pipeline=FakePipeline(), storage=storage)

    saved = service.save_text(
        user_id="u1",
        session_id="s1",
        content="保存这个菜谱：食材 鸡蛋，步骤1 打散。",
        content_type="recipe",
        title="鸡蛋菜谱",
    )

    assert saved.saved is True
    assert saved.content_type == "recipe"
    assert saved.storage == {"chunks": 1, "kv_pairs": 1, "triples": 1}
    chunk = storage.result.chunks[0]
    assert chunk.doc_category == DocCategory.RECIPE
    assert chunk.metadata["user_id"] == "u1"
    assert chunk.metadata["session_id"] == "s1"
    assert chunk.metadata["user_content_type"] == "recipe"
    assert chunk.metadata["visibility"] == "private"
    assert storage.result.kv_pairs[0].value["metadata"]["user_content_type"] == "recipe"
    assert storage.result.triples[0].properties["user_id"] == "u1"

