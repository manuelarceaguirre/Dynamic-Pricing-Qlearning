"""Data loading utilities."""
import pandas as pd
from pathlib import Path


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw electronic products data."""
    return pd.read_csv(path)
