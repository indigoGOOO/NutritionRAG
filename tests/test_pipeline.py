import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.index_pipeline import IndexingPipeline
from src.indexing.llm_client import BaseLLMClient
from src.indexing.models import SourceType


class MockLLMClient(BaseLLMClient):
    def generate(self, prompt: str, system: str = "") -> str:
        return "这是一个测试回答"

    def extract_structured(self, prompt: str, schema: dict, system: str = "") -> dict:
        return {
            "entities": [
                {
                    "name": "鸡蛋",
                    "type": "ingredient",
                    "attributes": {
                        "category": "蛋类",
                        "description": "常见食材",
                        "nutrition_tags": ["高蛋白"],
                        "contraindications": [],
                        "suitable_for": ["一般人群"],
                        "related_entities": ["牛奶"],
                    },
                }
            ],
            "triples": [
                {
                    "subject": "鸡蛋",
                    "subject_type": "ingredient",
                    "predicate": "contains",
                    "object": "蛋白质",
                    "object_type": "nutrient",
                    "properties": {"confidence": 0.9},
                }
            ],
        }


class MockVisionLLMClient(MockLLMClient):
    def __init__(self):
        self.image_calls = 0

    def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        system: str = "",
        image_mime: str = "image/jpeg",
    ) -> str:
        self.image_calls += 1
        assert image_base64
        assert image_mime == "image/png"
        return (
            '{"type": "natural", "subtype": "food_photo", "confidence": 0.95, '
            '"reason": "food photo", '
            '"description": "图片中是一份番茄炒蛋，包含鸡蛋和番茄，适合一般人群食用。", '
            '"foods": ["番茄", "鸡蛋"], '
            '"nutrition_notes": "蛋白质和维生素较丰富"}'
        )


class TestPipelineText:
    def setup_method(self):
        self.pipeline = IndexingPipeline(llm_client=MockLLMClient())

    def test_text_pipeline(self):
        text = "鸡蛋营养丰富，每100克含蛋白质13.3克，脂肪8.8克。适合一般人群食用。"

        result = self.pipeline.run_text(text, source_name="test")

        assert result.doc_id != ""
        assert len(result.chunks) > 0
        assert len(result.kv_pairs) > 0
        assert len(result.errors) == 0

    def test_recipe_text_pipeline(self):
        text = """番茄炒蛋

配料：番茄2个，鸡蛋3个，盐适量

步骤1：将鸡蛋打散
步骤2：番茄切块
步骤3：热锅炒蛋，加入番茄翻炒"""

        result = self.pipeline.run_text(text, source_name="recipe_test")

        assert result.doc_id != ""
        assert len(result.chunks) > 0

    def test_empty_text(self):
        result = self.pipeline.run_text("", source_name="empty")
        assert len(result.chunks) == 0

    def test_image_routes_to_single_doubao_vision_call_for_natural_scene(self, tmp_path):
        vision_llm = MockVisionLLMClient()
        pipeline = IndexingPipeline(llm_client=vision_llm)
        image_path = tmp_path / "meal.png"
        image_path.write_bytes(b"fake-image-bytes")

        document = pipeline.router.route(image_path)

        assert document.metadata.source_type == SourceType.IMAGE
        assert document.metadata.extra["image_type"] == "food_photo"
        assert document.metadata.extra["single_vision_call"] is True
        assert "番茄炒蛋" in document.text_content
        assert vision_llm.image_calls == 1
