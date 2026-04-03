#!/usr/bin/env bash
#
# Full pipeline: merge → dedup → train → evaluate
# Usage:
#   bash run_pipeline.sh                     # run all stages
#   bash run_pipeline.sh --skip-merge        # skip merge (reuse existing)
#   bash run_pipeline.sh --skip-dedup        # skip dedup
#   bash run_pipeline.sh --skip-train        # only data prep, no training
#   bash run_pipeline.sh --config path.yaml  # custom config for training
#
set -euo pipefail

# ===== Defaults =====
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MERGED_FILE="data/merged_styles.jsonl"
DEDUP_DIR="data/deduped_styles"
DEDUP_WORK_DIR="data/minhash_work"
RAW_DIR="data/raw"
CONFIG="configs/pipeline.yaml"

MIN_TEXT_LENGTH=20
DEDUP_WORKERS=4
DEDUP_BUCKETS=14
DEDUP_HASHES=8
DEDUP_NGRAMS=5

BALANCE="downsample"      # downsample | upsample | none

SKIP_MERGE=false
SKIP_DEDUP=false
SKIP_TRAIN=false

# ===== Parse args =====
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-merge)   SKIP_MERGE=true; shift ;;
        --skip-dedup)   SKIP_DEDUP=true; shift ;;
        --skip-train)   SKIP_TRAIN=true; shift ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --min-length)   MIN_TEXT_LENGTH="$2"; shift 2 ;;
        --workers)      DEDUP_WORKERS="$2"; shift 2 ;;
        --balance)      BALANCE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--skip-merge] [--skip-dedup] [--skip-train] [--config path] [--min-length N] [--workers N]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log() { echo -e "\n\033[1;36m[PIPELINE]\033[0m $(date '+%H:%M:%S') $*"; }

# ===== Stage 1: Merge datasets =====
if [ "$SKIP_MERGE" = false ]; then
    log "Stage 1/4 — Merging datasets..."
    python scripts/merge_datasets.py \
        --output "$MERGED_FILE" \
        --min-length "$MIN_TEXT_LENGTH" \
        --no-stratify
    log "Merged → $MERGED_FILE ($(wc -l < "$MERGED_FILE") lines)"
else
    log "Stage 1/4 — Skipped (--skip-merge). Using $MERGED_FILE"
fi

# ===== Stage 2: Dedup with datatrove =====
if [ "$SKIP_DEDUP" = false ]; then
    log "Stage 2/4 — MinHash deduplication (datatrove)..."

    # Clean previous run
    rm -rf "$DEDUP_DIR" "$DEDUP_WORK_DIR"

    python scripts/dedup_datatrove.py \
        --input "$MERGED_FILE" \
        --output "$DEDUP_DIR" \
        --work-dir "$DEDUP_WORK_DIR" \
        --workers "$DEDUP_WORKERS" \
        --num-buckets "$DEDUP_BUCKETS" \
        --hashes-per-bucket "$DEDUP_HASHES" \
        --n-grams "$DEDUP_NGRAMS"

    DEDUP_COUNT=$(zcat "$DEDUP_DIR"/*.jsonl.gz 2>/dev/null | wc -l || cat "$DEDUP_DIR"/*.jsonl 2>/dev/null | wc -l || echo 0)
    log "Deduped → $DEDUP_DIR ($DEDUP_COUNT lines)"
else
    log "Stage 2/4 — Skipped (--skip-dedup)"
fi

# ===== Stage 3: Prepare data for train.py =====
log "Stage 3/4 — Preparing data for training..."

mkdir -p "$RAW_DIR"
rm -f "$RAW_DIR"/*.jsonl

# Check if dedup output exists (even when --skip-dedup, prior run may have produced it)
FOUND_DEDUP=$(find "$DEDUP_DIR" -name "*.jsonl.gz" -o -name "*.jsonl" 2>/dev/null | head -1)

BALANCE_FLAG=""
if [ "$BALANCE" != "none" ]; then
    BALANCE_FLAG="--balance $BALANCE"
fi

if [ -n "$FOUND_DEDUP" ]; then
    python scripts/normalize_data.py \
        --input "$DEDUP_DIR" \
        --output "$RAW_DIR/data.jsonl" \
        --label-field style \
        $BALANCE_FLAG
    log "Normalized dedup output → $RAW_DIR/data.jsonl (balance=$BALANCE)"
else
    python scripts/normalize_data.py \
        --input "$MERGED_FILE" \
        --output "$RAW_DIR/data.jsonl" \
        --label-field style \
        $BALANCE_FLAG
    log "Normalized merged data → $RAW_DIR/data.jsonl (balance=$BALANCE)"
fi

FINAL_COUNT=$(wc -l < "$RAW_DIR/data.jsonl")
log "Training data ready: $FINAL_COUNT samples"

# ===== Stage 4: Train =====
if [ "$SKIP_TRAIN" = false ]; then
    log "Stage 4/4 — Training model..."
    python train.py --config "$CONFIG"
    log "Training complete! Model saved to outputs/"
else
    log "Stage 4/4 — Skipped (--skip-train)"
fi

log "Pipeline finished."
