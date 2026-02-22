from src.models.inference import MorphologyPredictor


class DummyBatch(dict):
    def to(self, _device):
        return self


class DummyTokenizer:
    def __call__(self, _word, return_tensors="pt", padding=True, truncation=True):
        return DummyBatch({"input_ids": [[1, 2, 3]]})

    def decode(self, _tokens, skip_special_tokens=True):
        return "الكلمة: كتاب\nالصنف الصرفي: اسم\nالجذر: كتب\nالحالة: معرب"


class DummyModel:
    device = "cpu"

    def generate(self, **_kwargs):
        return [[1, 2, 3]]


def test_predictor_smoke():
    predictor = MorphologyPredictor(model=DummyModel(), tokenizer=DummyTokenizer(), device="cpu")
    result = predictor.analyze_text("كتاب جميل")

    assert len(result) == 2
    assert result[0]["الكلمة"] == "كتاب"
    assert result[0]["الجذر"] == "كتب"

