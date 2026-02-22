import argparse
import json

from src.models.inference import load_predictor
from src.utils.logging import configure_logging
from src.utils.paths import resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run morphology inference.")
    parser.add_argument("--model-dir", default=str(resolve_path("trained_arabart_morph_model")))
    parser.add_argument("--text", required=True, help="Input word or sentence in Arabic.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging()
    predictor = load_predictor(args.model_dir)
    results = predictor.analyze_text(args.text)
    print(json.dumps({"result": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

