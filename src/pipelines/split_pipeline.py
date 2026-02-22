import argparse

from src.data.split import split_dataset
from src.utils.logging import configure_logging
from src.utils.paths import load_yaml, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split cleaned dataset into train/val/test.")
    parser.add_argument("--config", default=str(resolve_path("src/config/default.yaml")))
    parser.add_argument("--input-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--test-output-path")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging()
    cfg = load_yaml(args.config)
    paths_cfg = cfg.get("paths", {})

    split_dataset(
        input_path=args.input_path or paths_cfg.get(
            "cleaned_dataset", "morphological_descriptions_cleaned.csv"
        ),
        output_dir=args.output_dir or paths_cfg.get("split_dir", "data_ready"),
        test_output_path=args.test_output_path
        or paths_cfg.get("legacy_test_csv", "morph_test.csv"),
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

