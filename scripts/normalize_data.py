"""Normalize datatrove dedup output for training.

datatrove moves non-text fields into a nested "metadata" dict:
    {"text": "...", "metadata": {"style": "bao_chi", ...}}

This script flattens metadata back to top-level and keeps only
the fields needed for training (text + label).

Usage:
    python scripts/normalize_data.py --input data/deduped_styles --output data/raw/data.jsonl
    python scripts/normalize_data.py --input data/deduped_styles --output data/raw/data.jsonl --label-field style
"""

import argparse
import gzip
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
    # Extract text
    text = record.get("text", "")
    if not text:
        return None

    # Try label at top-level first
    label = record.get(label_field)

    # If not found, look inside metadata
    if label is None and "metadata" in record and isinstance(record["metadata"], dict):
        label = record["metadata"].get(label_field)

    if label is None:
        return None

    return {"text": text, label_field: str(label)}


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
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_files = find_input_files(input_path)
    if not input_files:
        logger.error("No JSONL files found in %s", input_path)
        return

    logger.info("Found %d input file(s): %s", len(input_files), [str(f) for f in input_files])

    total = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for fpath in input_files:
            logger.info("Processing %s ...", fpath)
            for record in iter_jsonl(fpath):
                normalized = normalize_record(record, args.label_field)
                if normalized is None:
                    skipped += 1
                    continue
                out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                total += 1

    logger.info("Done. Wrote %d records to %s (skipped %d)", total, output_path, skipped)


if __name__ == "__main__":
    main()
