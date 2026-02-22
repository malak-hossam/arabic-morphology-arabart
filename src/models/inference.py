import re
from dataclasses import dataclass
from typing import Dict, List, Optional


def extract_field(text: str, field_name: str) -> str:
    pattern = rf"{re.escape(field_name)}\s*:\s*(.*?)(?=\n\S+\s*:\s*|$)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def clean_root(root: str) -> str:
    return root if re.fullmatch(r"[اأإآء-ي]+", root or "") else "N/A"


@dataclass
class PredictionResult:
    word: str
    morph_type: str
    state: str
    root: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "الكلمة": self.word,
            "الصنف الصرفي": self.morph_type,
            "الحالة": self.state,
            "الجذر": self.root,
        }


class MorphologyPredictor:
    def __init__(
        self,
        model: object,
        tokenizer: object,
        device: str = "cpu",
        max_length: int = 128,
        num_beams: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams

    def _generate(self, word: str) -> str:
        encoded = self.tokenizer(
            word, return_tensors="pt", padding=True, truncation=True
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(self.device)

        outputs = self.model.generate(
            **encoded,
            max_length=self.max_length,
            num_beams=self.num_beams,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def predict_word(self, word: str) -> PredictionResult:
        decoded = self._generate(word)
        morph_type = extract_field(decoded, "الصنف الصرفي")
        state = extract_field(decoded, "الحالة")
        root = clean_root(extract_field(decoded, "الجذر"))
        return PredictionResult(word=word, morph_type=morph_type, state=state, root=root)

    def analyze_text(self, text: str) -> List[Dict[str, str]]:
        words = text.strip().split()
        return [self.predict_word(word).to_dict() for word in words]


def load_predictor(model_dir: str, device: Optional[str] = None) -> MorphologyPredictor:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers/torch are required for inference. Install requirements.txt."
        ) from exc

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(selected_device)
    return MorphologyPredictor(model=model, tokenizer=tokenizer, device=selected_device)

