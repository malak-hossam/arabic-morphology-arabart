# Data Card

## Dataset Assets
- Raw source: `MASAQ.csv`
- Cleaned seq2seq pairs: `morphological_descriptions_cleaned.csv`
- Train/validation/test splits:
  - `data_ready/train.csv`
  - `data_ready/val.csv`
  - `morph_test.csv` (legacy location) or `data_ready/test.csv`

## Provenance
- Source corpus: MASAQ (Morphologically-Analyzed and Syntactically-Annotated Quran data).
- The project consumes existing MASAQ-style morphological annotations and converts them into model-ready input-target pairs.

## Raw Schema (`MASAQ.csv`)
Observed columns:
- `ID`
- `Sura_No`
- `Verse_No`
- `Word_No`
- `Segment_No`
- `Word`
- `Without_Diacritics`
- `Segmented_Word`
- `Morph_Tag`
- `Morph_Type`
- `Punctuation_Mark`
- `Invariable_Declinable`
- `Syntactic_Role`
- `Possessive_Construct`
- `Case_Mood`
- `Case_Mood_Marker`
- `Phrase`
- `Phrasal_Function`
- `Gloss`

## Processed Schema (`morphological_descriptions_cleaned.csv`)
- `input`: reconstructed token
- `target`: multiline Arabic morphology description:
  - `الكلمة: ...`
  - `الصنف الصرفي: ...`
  - `الجذر: ...`
  - optional `الحالة: ...`

## Current Dataset Sizes (Repository Snapshot)
- Raw MASAQ rows: 157,676
- Cleaned seq2seq rows: 6,415
- Default split sizes from cleaned set:
  - Train: 5,132
  - Validation: 641
  - Test: 642

## Known Constraints
- Quranic domain bias may limit general-domain or dialect transfer.
- Text-generation target format can encode annotation inconsistencies.
- Root quality depends on analyzer availability and extraction heuristics.
- Some outputs may include malformed/non-Arabic roots; inference layer normalizes invalid roots to `N/A`.

## Data Handling Policy
- Large raw/processed datasets are intentionally ignored in `.gitignore`.
- Rebuild artifacts locally using pipeline commands documented in `README.md`.

