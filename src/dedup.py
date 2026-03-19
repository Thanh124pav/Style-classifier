"""MinHash-based deduplication using datatrove.

Provides both a pipeline-based approach (for large-scale data)
and a DataFrame-level wrapper for integration with existing code.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig

logger = logging.getLogger(__name__)


def deduplicate(
    df: pd.DataFrame,
    num_buckets: int = 14,
    hashes_per_bucket: int = 8,
    n_grams: int = 5,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Remove near-duplicate texts using datatrove MinHash pipeline.

    Args:
        df: DataFrame with 'text' column.
        num_buckets: Number of LSH buckets.
        hashes_per_bucket: Number of hashes per bucket.
        n_grams: Size of character n-grams for shingling.
        num_workers: Number of parallel workers.

    Returns:
        Deduplicated DataFrame.
    """
    original_size = len(df)
    logger.info("Starting deduplication on %d records...", original_size)

    tmpdir = tempfile.mkdtemp(prefix="dedup_")
    try:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        work_dir = Path(tmpdir) / "work"
        input_dir.mkdir()

        # Write DataFrame to JSONL for datatrove
        input_file = input_dir / "data.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                record = row.to_dict()
                # datatrove expects a "text" key
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        minhash_config = MinhashConfig(
            hash_config=HashConfig(precision=64),
            num_buckets=num_buckets,
            hashes_per_bucket=hashes_per_bucket,
            n_grams=n_grams,
        )

        reader = JsonlReader(data_folder=str(input_dir), text_key="text")

        # Stage 1: Signatures
        LocalPipelineExecutor(
            pipeline=[reader, MinhashDedupSignature(output_folder=f"{work_dir}/sigs", config=minhash_config)],
            tasks=num_workers,
            logging_dir=f"{work_dir}/logs/s1",
        ).run()

        # Stage 2: Buckets
        LocalPipelineExecutor(
            pipeline=[MinhashDedupBuckets(input_folder=f"{work_dir}/sigs", output_folder=f"{work_dir}/buckets", config=minhash_config)],
            tasks=minhash_config.num_buckets,
            logging_dir=f"{work_dir}/logs/s2",
        ).run()

        # Stage 3: Cluster
        LocalPipelineExecutor(
            pipeline=[MinhashDedupCluster(input_folder=f"{work_dir}/buckets", output_folder=f"{work_dir}/clusters", config=minhash_config)],
            tasks=1,
            logging_dir=f"{work_dir}/logs/s3",
        ).run()

        # Stage 4: Filter
        LocalPipelineExecutor(
            pipeline=[
                reader,
                MinhashDedupFilter(input_folder=f"{work_dir}/clusters"),
                JsonlWriter(str(output_dir)),
            ],
            tasks=num_workers,
            logging_dir=f"{work_dir}/logs/s4",
        ).run()

        # Read back deduplicated results
        records = []
        for jsonl_file in sorted(output_dir.rglob("*.jsonl")):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

        df_dedup = pd.DataFrame(records)
        # Keep only original columns
        df_dedup = df_dedup[[c for c in df.columns if c in df_dedup.columns]].reset_index(drop=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    removed = original_size - len(df_dedup)
    logger.info(
        "Deduplication complete. Removed %d duplicates (%.1f%%). %d records remaining.",
        removed, removed / original_size * 100 if original_size else 0, len(df_dedup),
    )
    return df_dedup
