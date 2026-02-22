import argparse

from src.data.preprocess import prepare_dataset
from src.utils.logging import configure_logging
from src.utils.paths import load_yaml, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare morphology training dataset.")
    parser.add_argument("--config", default=str(resolve_path("src/config/default.yaml")))
    parser.add_argument("--raw-path")
    parser.add_argument("--output-path")
    parser.add_argument("--use-camel-tools", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging()
    cfg = load_yaml(args.config)
    paths_cfg = cfg.get("paths", {})

    raw_path = args.raw_path or paths_cfg.get("raw_dataset", "MASAQ.csv")
    output_path = args.output_path or paths_cfg.get(
        "cleaned_dataset", "morphological_descriptions_cleaned.csv"
    )

    prepare_dataset(
        raw_path=raw_path,
        output_path=output_path,
        use_camel_tools=args.use_camel_tools,
    )


if __name__ == "__main__":
    main()

