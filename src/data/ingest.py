import pandas as pd

from src.data.schema import RAW_REQUIRED_COLUMNS
from src.utils.paths import resolve_path


def read_raw_masaq(path: str) -> pd.DataFrame:
    csv_path = resolve_path(path)
    df = pd.read_csv(csv_path, low_memory=False)

    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def read_training_pairs(path: str) -> pd.DataFrame:
    csv_path = resolve_path(path)
    return pd.read_csv(csv_path)

