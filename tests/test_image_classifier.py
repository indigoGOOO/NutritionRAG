import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.image_classifier import ImageClassifier


def test_strong_ocr_probe_detects_nutrition_table(monkeypatch, tmp_path):
    image_path = tmp_path / "ambiguous.png"
    image_path.write_bytes(b"fake-image-bytes" * 10000)
    classifier = ImageClassifier()

    monkeypatch.setattr(
        classifier,
        "_extract_ocr_text",
        lambda path: "营养成分表 每100g 蛋白质 12g 脂肪 3g 碳水化合物 20g NRV%",
    )

    decision = classifier.decide_image_route(image_path)

    assert decision.kind == "document"
    assert decision.subtype == "nutrition_table"
    assert decision.decision == "direct_document"
    assert decision.signals["ocr_hit_count"] >= 3


def test_weak_ocr_probe_requires_vision(monkeypatch, tmp_path):
    image_path = tmp_path / "ambiguous.png"
    image_path.write_bytes(b"fake-image-bytes" * 10000)
    classifier = ImageClassifier()

    monkeypatch.setattr(classifier, "_extract_ocr_text", lambda path: "蛋白质")

    decision = classifier.decide_image_route(image_path)

    assert decision.kind == "document"
    assert decision.subtype == "nutrition_table"
    assert decision.decision == "vision_required"
    assert decision.reason == "ocr_weak_document_hint"


def test_ocr_probe_failure_keeps_ambiguous_image_for_vision(tmp_path):
    image_path = tmp_path / "ambiguous.png"
    image_path.write_bytes(b"fake-image-bytes" * 10000)
    classifier = ImageClassifier()

    decision = classifier.decide_image_route(image_path)

    assert decision.kind == "unknown"
    assert decision.subtype == "unknown"
    assert decision.decision == "vision_required"
    assert decision.confidence == 0.35


def test_filename_document_short_circuits_ocr(monkeypatch, tmp_path):
    image_path = tmp_path / "nutrition_label.png"
    image_path.write_bytes(b"fake")
    classifier = ImageClassifier()

    def fail_if_called(path):
        raise AssertionError("filename document match should skip OCR")

    monkeypatch.setattr(classifier, "_extract_ocr_text", fail_if_called)

    decision = classifier.decide_image_route(image_path)

    assert decision.kind == "document"
    assert decision.subtype == "nutrition_table"
    assert decision.decision == "direct_document"
    assert decision.confidence == 0.9
