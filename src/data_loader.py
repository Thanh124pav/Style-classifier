"""Data loading module for JSONL files."""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> list[dict]:
    """Load a single JSONL file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line {line_num} in {file_path}")
    logger.info(f"Loaded {len(records)} records from {file_path}")
    return records


def load_data(
    input_path: str,
    text_field: str = "text",
    label_field: str = "label",
) -> pd.DataFrame:
    """Load data from a JSONL file or a folder of JSONL files.

    Args:
        input_path: Path to a single JSONL file or directory containing JSONL files.
        text_field: Name of the field containing the text.
        label_field: Name of the field containing the label.

    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    path = Path(input_path)
    all_records = []

    if path.is_file() and path.suffix in (".jsonl", ".json"):
        all_records = load_jsonl(str(path))
    elif path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {input_path}")
        for f in jsonl_files:
            all_records.extend(load_jsonl(str(f)))
    else:
        raise FileNotFoundError(f"Invalid input path: {input_path}")

    if not all_records:
        raise ValueError("No records loaded from input data.")

    df = pd.DataFrame(all_records)

    # datatrove nests fields inside "metadata" after dedup — extract them
    if "metadata" in df.columns:
        meta_df = pd.json_normalize(df["metadata"])
        for col in meta_df.columns:
            if col not in df.columns:
                df[col] = meta_df[col]
        df = df.drop(columns=["metadata"])
        logger.info(f"Extracted metadata fields: {list(meta_df.columns)}")

    # Validate required fields
    if text_field not in df.columns:
        raise ValueError(f"Text field '{text_field}' not found. Columns: {list(df.columns)}")
    if label_field not in df.columns:
        raise ValueError(f"Label field '{label_field}' not found. Columns: {list(df.columns)}")

    df = df.rename(columns={text_field: "text", label_field: "label"})
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    logger.info(f"Total records loaded: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    from sklearn.model_selection import train_test_split

    stratify_col = df["label"] if stratify else None

    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify_col
    )

    # Second split: train vs val
    relative_val_size = val_size / (1 - test_size)
    stratify_col = train_val["label"] if stratify else None
    train, val = train_test_split(
        train_val, test_size=relative_val_size, random_state=seed, stratify=stratify_col
    )

    logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
