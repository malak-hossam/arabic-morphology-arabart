import pandas as pd

from src.data.preprocess import build_training_pairs, reconstruct_word_level_dataframe


def test_reconstruct_word_level_dataframe():
    raw = pd.DataFrame(
        [
            {
                "ID": 1,
                "Word_No": 1,
                "Segmented_Word": "ب",
                "Morph_Tag": "PREP",
                "Morph_Type": "Prefix",
                "Invariable_Declinable": "INVAR",
            },
            {
                "ID": 1,
                "Word_No": 1,
                "Segmented_Word": "اسم",
                "Morph_Tag": "NOUN_ABSTRACT",
                "Morph_Type": "Stem",
                "Invariable_Declinable": "DECLN",
            },
            {
                "ID": 1,
                "Word_No": 1,
                "Segmented_Word": "ه",
                "Morph_Tag": "PRON",
                "Morph_Type": "Suffix",
                "Invariable_Declinable": "INVAR",
            },
        ]
    )

    merged = reconstruct_word_level_dataframe(raw)
    assert len(merged) == 1
    assert merged.loc[0, "Full_Word"] == "باسمه"


def test_build_training_pairs_with_custom_root():
    raw = pd.DataFrame(
        [
            {
                "ID": 10,
                "Word_No": 3,
                "Segmented_Word": "ال",
                "Morph_Tag": "DET",
                "Morph_Type": "Prefix",
                "Invariable_Declinable": "INVAR",
            },
            {
                "ID": 10,
                "Word_No": 3,
                "Segmented_Word": "كتاب",
                "Morph_Tag": "NOUN_ABSTRACT",
                "Morph_Type": "Stem",
                "Invariable_Declinable": "DECLN",
            },
        ]
    )

    pairs = build_training_pairs(raw, use_camel_tools=False, root_fn=lambda _: "كتب")
    assert list(pairs.columns) == ["input", "target"]
    assert len(pairs) == 1
    assert "الجذر: كتب" in pairs.iloc[0]["target"]

