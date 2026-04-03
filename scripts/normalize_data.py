"""Normalize datatrove dedup output for training.

datatrove moves non-text fields into a nested "metadata" dict:
    {"text": "...", "metadata": {"style": "bao_chi", ...}}

This script:
  1. Flattens metadata back to top-level
  2. Cleans Vietnamese text (URLs, HTML, unicode, etc.)
  3. Optionally performs Vietnamese word segmentation (underthesea)
  4. Filters empty / too-short texts
  5. Optionally balances classes (downsample / upsample)
  6. Writes clean JSONL ready for training

Usage:
    python scripts/normalize_data.py -i data/deduped_styles -o data/raw/data.jsonl
    python scripts/normalize_data.py -i data/deduped_styles -o data/raw/data.jsonl --balance downsample
    python scripts/normalize_data.py -i data/deduped_styles -o data/raw/data.jsonl --word-segment --balance upsample
"""

import argparse
import gzip
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Import cleaning functions from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import clean_vietnamese_text, word_segment


def iter_jsonl(path: Path):
    """Yield parsed JSON objects from a .jsonl or .jsonl.gz file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", line_num, path)


def normalize_record(record: dict, label_field: str) -> dict | None:
    """Flatten metadata and extract text + label."""
    text = record.get("text", "")
    if not text:
        return None

    label = record.get(label_field)
    if label is None and "metadata" in record and isinstance(record["metadata"], dict):
        label = record["metadata"].get(label_field)

    if label is None:
        return None

    return {"text": text, label_field: str(label)}


def clean_record(args: tuple) -> dict | None:
    """Clean a single record's text. Used by multiprocessing pool."""
    record, do_word_seg, min_length = args
    text = record["text"]

    text = clean_vietnamese_text(text)

    if do_word_seg:
        text = word_segment(text)

    if len(text) < min_length:
        return None

    record["text"] = text
    return record


def clean_records_parallel(
    records: list[dict],
    do_word_seg: bool = False,
    min_length: int = 10,
    num_workers: int = 4,
) -> list[dict]:
    """Clean all records in parallel using multiprocessing."""
    logger.info(
        "Cleaning %d records (word_segment=%s, min_length=%d, workers=%d)...",
        len(records), do_word_seg, min_length, num_workers,
    )

    tasks = [(r, do_word_seg, min_length) for r in records]

    if num_workers <= 1:
        results = [clean_record(t) for t in tasks]
    else:
        with Pool(num_workers) as pool:
            results = pool.map(clean_record, tasks, chunksize=1000)

    cleaned = [r for r in results if r is not None]
    logger.info("Cleaning done. %d → %d records (removed %d empty/short)",
                len(records), len(cleaned), len(records) - len(cleaned))
    return cleaned


def find_input_files(input_path: Path) -> list[Path]:
    """Find all JSONL/JSONL.GZ files in input path."""
    if input_path.is_file():
        return [input_path]

    files = sorted(input_path.glob("**/*.jsonl.gz")) + sorted(input_path.glob("**/*.jsonl"))
    # Deduplicate (avoid matching .jsonl within .jsonl.gz paths)
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def balance_records(
    records: list[dict],
    label_field: str,
    method: str,
    seed: int = 42,
) -> list[dict]:
    """Balance class distribution.

    Args:
        records: List of normalized record dicts.
        label_field: Name of the label field.
        method: 'downsample' (cut to min class), 'upsample' (repeat to max class).
        seed: Random seed.

    Returns:
        Balanced list of records.
    """
    rng = random.Random(seed)

    # Group by label
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_label[r[label_field]].append(r)

    counts = {k: len(v) for k, v in by_label.items()}
    logger.info("--- Class distribution before balancing ---")
    for label, count in sorted(counts.items()):
        logger.info("  %s: %d", label, count)

    if method == "downsample":
        target = min(counts.values())
        logger.info("Downsampling all classes to %d", target)
        balanced = []
        for label, group in by_label.items():
            balanced.extend(rng.sample(group, min(target, len(group))))
    elif method == "upsample":
        target = max(counts.values())
        logger.info("Upsampling all classes to %d", target)
        balanced = []
        for label, group in by_label.items():
            if len(group) >= target:
                balanced.extend(rng.sample(group, target))
            else:
                # Keep all originals + sample extra with replacement
                balanced.extend(group)
                extra = target - len(group)
                balanced.extend(rng.choices(group, k=extra))
    else:
        raise ValueError(f"Unknown balance method: {method}")

    rng.shuffle(balanced)

    result_counts = Counter(r[label_field] for r in balanced)
    logger.info("--- Class distribution after balancing ---")
    for label, count in sorted(result_counts.items()):
        logger.info("  %s: %d", label, count)
    logger.info("Total: %d", len(balanced))

    return balanced


def main():
    parser = argparse.ArgumentParser(description="Normalize datatrove dedup output for training")
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input file or directory (datatrove output with .jsonl/.jsonl.gz)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--label-field", default="style",
        help="Name of the label field to extract (default: style)",
    )
    # Cleaning options
    parser.add_argument(
        "--word-segment", action="store_true",
        help="Apply Vietnamese word segmentation (underthesea). Slow but improves quality.",
    )
    parser.add_argument(
        "--min-length", type=int, default=10,
        help="Drop texts shorter than this after cleaning (default: 10 chars)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers for text cleaning (default: 4)",
    )
    # Balance options
    parser.add_argument(
        "--balance", choices=["downsample", "upsample"], default=None,
        help="Balance strategy: downsample (cut to smallest class) or upsample (repeat to largest class)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balancing")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_files = find_input_files(input_path)
    if not input_files:
        logger.error("No JSONL files found in %s", input_path)
        return

    logger.info("Found %d input file(s): %s", len(input_files), [str(f) for f in input_files])

    # 1. Read and normalize (flatten metadata)
    records = []
    skipped = 0
    for fpath in input_files:
        logger.info("Processing %s ...", fpath)
        for record in iter_jsonl(fpath):
            normalized = normalize_record(record, args.label_field)
            if normalized is None:
                skipped += 1
                continue
            records.append(normalized)

    logger.info("Normalized %d records (skipped %d)", len(records), skipped)

    # 2. Clean text (parallel)
    records = clean_records_parallel(
        records,
        do_word_seg=args.word_segment,
        min_length=args.min_length,
        num_workers=args.workers,
    )

    # 3. Balance if requested
    if args.balance:
        records = balance_records(records, args.label_field, args.balance, args.seed)

    # 4. Write output
    with open(output_path, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Done. Wrote %d records to %s", len(records), output_path)


if __name__ == "__main__":
    main()
