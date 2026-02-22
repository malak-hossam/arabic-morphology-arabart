import argparse
import json

import pandas as pd

from src.models.inference import load_predictor
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import resolve_path


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained model on held-out test CSV.")
    parser.add_argument("--model-dir", default=str(resolve_path("trained_arabart_morph_model")))
    parser.add_argument("--test-csv", default=str(resolve_path("morph_test.csv")))
    parser.add_argument("--max-samples", type=int, default=100)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging()

    df = pd.read_csv(args.test_csv)
    if "input" not in df.columns or "target" not in df.columns:
        raise ValueError("Test CSV must contain `input` and `target` columns.")

    predictor = load_predictor(args.model_dir)
    samples = df.head(args.max_samples)

    exact_match_hits = 0
    for _, row in samples.iterrows():
        pred = predictor.predict_word(str(row["input"]))
        pred_text = "\n".join(
            [
                f"الكلمة: {pred.word}",
                f"الصنف الصرفي: {pred.morph_type}",
                f"الجذر: {pred.root}",
                f"الحالة: {pred.state}",
            ]
        ).strip()
        if pred_text == str(row["target"]).strip():
            exact_match_hits += 1

    metrics = {
        "evaluated_samples": int(len(samples)),
        "exact_match": (exact_match_hits / len(samples)) if len(samples) else 0.0,
    }
    LOGGER.info("Evaluation metrics: %s", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

