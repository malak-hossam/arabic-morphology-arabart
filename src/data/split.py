import argparse
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logging import configure_logging, get_logger
from src.utils.paths import ensure_parent, resolve_path


LOGGER = get_logger(__name__)


@dataclass
class SplitResult:
    train_size: int
    val_size: int
    test_size: int


def split_dataset(
    input_path: str,
    output_dir: str,
    test_output_path: str | None = None,
    random_state: int = 42,
) -> SplitResult:
    df = pd.read_csv(resolve_path(input_path)).dropna().drop_duplicates()

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)

    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    if test_output_path:
        test_path = ensure_parent(test_output_path)
    else:
        test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    LOGGER.info("Saved train split to %s (%d rows)", train_path, len(train_df))
    LOGGER.info("Saved val split to %s (%d rows)", val_path, len(val_df))
    LOGGER.info("Saved test split to %s (%d rows)", test_path, len(test_df))

    return SplitResult(
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split cleaned morphology dataset.")
    parser.add_argument(
        "--input-path",
        default=str(resolve_path("morphological_descriptions_cleaned.csv")),
    )
    parser.add_argument("--output-dir", default=str(resolve_path("data_ready")))
    parser.add_argument(
        "--test-output-path",
        default=str(resolve_path("morph_test.csv")),
        help="Legacy-compatible default. Set empty string to save test.csv in output-dir.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging()
    split_dataset(
        input_path=args.input_path,
        output_dir=args.output_dir,
        test_output_path=args.test_output_path or None,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

