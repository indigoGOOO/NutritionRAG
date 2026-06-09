import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.document_router import DocumentRouter
from src.indexing.llm_client import BaseLLMClient


class VisionLLM(BaseLLMClient):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, system: str = "") -> str:
        return ""

    def extract_structured(self, prompt: str, schema: dict, system: str = "") -> dict:
        return {}

    def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        system: str = "",
        image_mime: str = "image/jpeg",
    ) -> str:
        self.calls += 1
        return (
            '{"type": "natural", "subtype": "food_photo", "confidence": 0.91, '
            '"reason": "meal photo", "description": "A plate of tomato eggs.", '
            '"foods": ["tomato", "egg"], "nutrition_notes": "protein and vitamins"}'
        )


def test_router_attaches_routing_trace_for_vision_image(tmp_path):
    image_path = tmp_path / "meal.png"
    image_path.write_bytes(b"fake-image-bytes")
    llm = VisionLLM()
    router = DocumentRouter(llm_client=llm)

    document = router.route(image_path)

    trace = document.metadata.extra["routing_trace"]
    assert trace["lightweight"]["decision"] == "vision_required"
    assert trace["lightweight"]["reason"] == "filename_natural_hint"
    assert trace["vision"]["kind"] == "natural"
    assert trace["vision"]["subtype"] == "food_photo"
    assert llm.calls == 1
