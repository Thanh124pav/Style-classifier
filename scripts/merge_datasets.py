"""Merge multiple datasets into a single JSONL file with stratified sampling."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
DATASETS = [
    {
        "path": "/storage-nlp/nlp/dungdx4/tmp/Vietnamese/thanhpv43/process_data_thanh/original_data/tla.jsonl",
        "text_field": "content",
        "style": "hanh_chinh",
    },
    {
        "path": "/raid/models/models_rsync/datasets/fb_comment_10m.jsonl",
        "text_field": "content",
        "style": "sinh_hoat",
    },
    {
        "path": "/storage-nlp/nlp/dungdx4/tmp/Vietnamese/thanhpv43/process_data_thanh/original_data/databaochi.jsonl",
        "text_field": "content",
        "style": "bao_chi",
    },
    {
        "path": "/raid/models/models_rsync/hapv14/functional_style_datasets/functionalstyle-labels_260317.csv",
        "text_field": "text",
        "style_field": "style",  # CSV already has style column
    },
]

MIN_TEXT_LENGTH = 20  # minimum characters to keep a sample


def load_jsonl(path: str, text_field: str, style: str) -> pd.DataFrame:
    """Load a JSONL file, extract text and assign a fixed style label."""
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s:%d", path, i + 1)
                continue
            text = obj.get(text_field, "")
            if text:
                records.append({"text": text, "style": style})
    df = pd.DataFrame(records)
    logger.info("Loaded %d rows from %s (style=%s)", len(df), path, style)
    return df


STYLE_NAME_MAP = {
    "chính luận": "chinh_luan",
    "khoa học": "khoa_hoc",
}


def load_csv(path: str, text_field: str, style_field: str) -> pd.DataFrame:
    """Load a CSV file that already contains text and style columns."""
    df = pd.read_csv(path, usecols=[text_field, style_field])
    df = df.rename(columns={text_field: "text", style_field: "style"})
    df = df.dropna(subset=["text", "style"])
    df["style"] = df["style"].str.strip().replace(STYLE_NAME_MAP)
    logger.info("Loaded %d rows from %s (styles: %s)", len(df), path, df["style"].unique().tolist())
    return df


def filter_short(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    """Remove samples shorter than min_len characters."""
    before = len(df)
    df = df[df["text"].str.len() >= min_len].reset_index(drop=True)
    logger.info("Filtered short texts (<%d chars): %d -> %d", min_len, before, len(df))
    return df


def stratify_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Down-sample each style to the size of the smallest class for balance."""
    counts = df["style"].value_counts()
    logger.info("--- Class distribution before stratification ---")
    for style, count in counts.items():
        logger.info("  %s: %d", style, count)

    min_count = counts.min()
    logger.info("Stratifying to %d samples per class", min_count)

    balanced = (
        df.groupby("style", group_keys=False)
        .apply(lambda g: g.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )

    logger.info("--- Class distribution after stratification ---")
    for style, count in balanced["style"].value_counts().items():
        logger.info("  %s: %d", style, count)
    logger.info("Total: %d", len(balanced))

    return balanced


def save_jsonl(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({"text": row["text"], "style": row["style"]}, ensure_ascii=False) + "\n")
    logger.info("Saved %d samples to %s", len(df), path)


def main():
    parser = argparse.ArgumentParser(description="Merge and stratify style classification datasets")
    parser.add_argument(
        "-o", "--output",
        default="data/merged_styles.jsonl",
        help="Output JSONL path (default: data/merged_styles.jsonl)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=MIN_TEXT_LENGTH,
        help=f"Minimum text length in characters (default: {MIN_TEXT_LENGTH})",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Skip stratified balancing",
    )
    args = parser.parse_args()

    # 1. Load all datasets
    frames = []
    for ds in DATASETS:
        path = ds["path"]
        if path.endswith(".csv"):
            frames.append(load_csv(path, ds["text_field"], ds["style_field"]))
        else:
            frames.append(load_jsonl(path, ds["text_field"], ds["style"]))

    df = pd.concat(frames, ignore_index=True)
    logger.info("Total merged: %d samples", len(df))

    # 2. Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info("Removed %d duplicates -> %d", before - len(df), len(df))

    # 3. Filter short texts
    df = filter_short(df, args.min_length)

    # 4. Stratified balancing
    if not args.no_stratify:
        df = stratify_balance(df)

    # 5. Shuffle and save
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    save_jsonl(df, args.output)


if __name__ == "__main__":
    main()
