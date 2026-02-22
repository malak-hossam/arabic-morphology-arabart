import argparse
import re
from typing import Callable, Optional

import pandas as pd

from src.data.ingest import read_raw_masaq
from src.data.morph_maps import DECLINABILITY_AR, MORPH_TAG_AR
from src.data.schema import TRAIN_COLUMNS
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import ensure_parent, resolve_path


LOGGER = get_logger(__name__)


class RootExtractor:
    def __init__(self, use_camel_tools: bool = True):
        self._analyzer = None
        if use_camel_tools:
            self._analyzer = self._init_camel_analyzer()

    @staticmethod
    def _init_camel_analyzer():
        try:
            from camel_tools.morphology.analyzer import Analyzer
            from camel_tools.morphology.database import MorphologyDB
        except ImportError:
            LOGGER.warning(
                "camel_tools is not installed; falling back to heuristic root extraction."
            )
            return None

        try:
            db = MorphologyDB.builtin_db()
            return Analyzer(db)
        except Exception:
            LOGGER.warning(
                "Failed to initialize CAMeL analyzer; using heuristic root extraction."
            )
            return None

    @staticmethod
    def _heuristic_root(word: str) -> str:
        letters = re.findall(r"[اأإآء-ي]", str(word))
        if not letters:
            return ""
        if len(letters) >= 3:
            return "".join(letters[:3])
        return "".join(letters)

    def extract(self, word: str) -> str:
        if not self._analyzer:
            return self._heuristic_root(word)

        try:
            analyses = self._analyzer.analyze(word)
            for analysis in analyses:
                root = analysis.get("root", "")
                if root and "#" not in root and root != word and len(root) <= len(word):
                    return root
        except Exception:
            return self._heuristic_root(word)

        return self._heuristic_root(word)


def reconstruct_word_level_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    stems = (
        raw_df[raw_df["Morph_Type"] == "Stem"]
        .groupby(["ID", "Word_No"])[["Segmented_Word", "Morph_Tag", "Invariable_Declinable"]]
        .first()
        .reset_index()
    )
    prefixes = (
        raw_df[raw_df["Morph_Type"] == "Prefix"]
        .groupby(["ID", "Word_No"])["Segmented_Word"]
        .apply(lambda x: "".join(x.dropna().astype(str)))
        .reset_index(name="prefix")
    )
    suffixes = (
        raw_df[raw_df["Morph_Type"] == "Suffix"]
        .groupby(["ID", "Word_No"])["Segmented_Word"]
        .apply(lambda x: "".join(x.dropna().astype(str)))
        .reset_index(name="suffix")
    )

    merged = stems.merge(prefixes, on=["ID", "Word_No"], how="left").merge(
        suffixes, on=["ID", "Word_No"], how="left"
    )
    merged = merged.fillna("")

    merged["Full_Word"] = merged["prefix"] + merged["Segmented_Word"] + merged["suffix"]
    merged["Declinability"] = merged["Invariable_Declinable"]
    return merged


def generate_target(
    word: str,
    root: str,
    morph_tag: str,
    declinability: str,
    morph_map: Optional[dict] = None,
    decl_map: Optional[dict] = None,
) -> str:
    morph_map = morph_map or MORPH_TAG_AR
    decl_map = decl_map or DECLINABILITY_AR

    tag_ar = morph_map.get(morph_tag, morph_tag)
    decl_ar = decl_map.get(declinability, "")

    lines = [f"الكلمة: {word}", f"الصنف الصرفي: {tag_ar}", f"الجذر: {root}"]
    if decl_ar:
        lines.append(f"الحالة: {decl_ar}")
    return "\n".join(lines)


def build_training_pairs(
    raw_df: pd.DataFrame,
    use_camel_tools: bool = True,
    root_fn: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    merged = reconstruct_word_level_dataframe(raw_df)
    extractor = RootExtractor(use_camel_tools=use_camel_tools)
    root_function = root_fn or extractor.extract

    merged["Root"] = merged["Full_Word"].apply(root_function)

    filtered = merged[
        (merged["Root"] != "")
        & (~merged["Root"].astype(str).str.contains("#"))
        & (merged["Root"] != merged["Full_Word"])
        & (merged["Root"].astype(str).str.len() <= merged["Full_Word"].astype(str).str.len())
    ].copy()

    filtered["input"] = filtered["Full_Word"]
    filtered["target"] = filtered.apply(
        lambda row: generate_target(
            word=row["Full_Word"],
            root=row["Root"],
            morph_tag=row["Morph_Tag"],
            declinability=row["Declinability"],
        ),
        axis=1,
    )

    final_df = filtered[TRAIN_COLUMNS].drop_duplicates().reset_index(drop=True)
    return final_df


def prepare_dataset(
    raw_path: str,
    output_path: str,
    use_camel_tools: bool = True,
) -> pd.DataFrame:
    raw_df = read_raw_masaq(raw_path)
    final_df = build_training_pairs(raw_df, use_camel_tools=use_camel_tools)

    output_csv = ensure_parent(output_path)
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    LOGGER.info("Prepared dataset saved to %s", output_csv)
    LOGGER.info("Number of samples: %d", len(final_df))
    return final_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MASAQ morphology dataset.")
    parser.add_argument("--raw-path", default=str(resolve_path("MASAQ.csv")))
    parser.add_argument(
        "--output-path",
        default=str(resolve_path("morphological_descriptions_cleaned.csv")),
    )
    parser.add_argument(
        "--use-camel-tools",
        action="store_true",
        help="Use CAMeL Tools for root extraction if available.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging()
    prepare_dataset(
        raw_path=args.raw_path,
        output_path=args.output_path,
        use_camel_tools=args.use_camel_tools,
    )


if __name__ == "__main__":
    main()

