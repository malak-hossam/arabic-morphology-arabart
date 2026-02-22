from typing import Tuple


def load_model_and_tokenizer(model_name_or_path: str) -> Tuple[object, object]:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install dependencies from requirements.txt."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    return model, tokenizer


def save_model_and_tokenizer(model: object, tokenizer: object, output_dir: str) -> None:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

