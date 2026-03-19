"""MinHash-based deduplication using datatrove (HuggingFace).

Runs a 4-stage local pipeline:
  1. Compute MinHash signatures
  2. Find bucket matches (LSH)
  3. Cluster duplicates
  4. Filter and write deduplicated output
"""

import argparse
import logging
from pathlib import Path

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "data/merged_styles.jsonl"
DEFAULT_OUTPUT = "data/deduped_styles"
DEFAULT_WORK_DIR = "data/minhash_work"


def run_dedup(
    input_path: str,
    output_dir: str,
    work_dir: str,
    num_workers: int = 4,
    num_buckets: int = 14,
    hashes_per_bucket: int = 8,
    n_grams: int = 5,
):
    """Run the 4-stage MinHash dedup pipeline locally."""

    input_path = str(Path(input_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    work_dir = str(Path(work_dir).resolve())

    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        num_buckets=num_buckets,
        hashes_per_bucket=hashes_per_bucket,
        n_grams=n_grams,
    )

    input_reader = JsonlReader(
        data_folder=str(Path(input_path).parent),
        glob_pattern=Path(input_path).name,
        text_key="text",
    )

    # Stage 1: Compute MinHash signatures
    logger.info("=== Stage 1/4: Computing MinHash signatures ===")
    stage1 = LocalPipelineExecutor(
        pipeline=[
            input_reader,
            MinhashDedupSignature(
                output_folder=f"{work_dir}/signatures",
                config=minhash_config,
            ),
        ],
        tasks=num_workers,
        logging_dir=f"{work_dir}/logs/stage1",
    )
    stage1.run()

    # Stage 2: Find bucket matches (LSH)
    logger.info("=== Stage 2/4: Finding bucket matches ===")
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{work_dir}/signatures",
                output_folder=f"{work_dir}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        logging_dir=f"{work_dir}/logs/stage2",
    )
    stage2.run()

    # Stage 3: Cluster duplicates
    logger.info("=== Stage 3/4: Clustering duplicates ===")
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{work_dir}/buckets",
                output_folder=f"{work_dir}/clusters",
                config=minhash_config,
            ),
        ],
        tasks=1,
        logging_dir=f"{work_dir}/logs/stage3",
    )
    stage3.run()

    # Stage 4: Filter duplicates and write output
    logger.info("=== Stage 4/4: Filtering duplicates and writing output ===")
    stage4 = LocalPipelineExecutor(
        pipeline=[
            input_reader,
            MinhashDedupFilter(
                input_folder=f"{work_dir}/clusters",
                exclusion_writer=JsonlWriter(f"{work_dir}/removed"),
            ),
            JsonlWriter(output_dir),
        ],
        tasks=num_workers,
        logging_dir=f"{work_dir}/logs/stage4",
    )
    stage4.run()

    logger.info("=== Deduplication complete! Output: %s ===", output_dir)


def main():
    parser = argparse.ArgumentParser(description="MinHash dedup with datatrove")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="Input JSONL path")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("-w", "--work-dir", default=DEFAULT_WORK_DIR, help="Working directory for intermediate files")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--num-buckets", type=int, default=14, help="Number of LSH buckets")
    parser.add_argument("--hashes-per-bucket", type=int, default=8, help="Hashes per bucket")
    parser.add_argument("--n-grams", type=int, default=5, help="N-gram size for shingling")
    args = parser.parse_args()

    run_dedup(
        input_path=args.input,
        output_dir=args.output,
        work_dir=args.work_dir,
        num_workers=args.workers,
        num_buckets=args.num_buckets,
        hashes_per_bucket=args.hashes_per_bucket,
        n_grams=args.n_grams,
    )


if __name__ == "__main__":
    main()
