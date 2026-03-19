"""MinHash-based deduplication module using datasketch."""

import logging

import pandas as pd
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)


def get_ngrams(text: str, n: int = 5) -> list[str]:
    """Extract character-level n-grams from text."""
    text = text.strip()
    if len(text) < n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def build_minhash(text: str, num_perm: int = 128, ngram_size: int = 5) -> MinHash:
    """Build a MinHash signature for a text."""
    m = MinHash(num_perm=num_perm)
    for gram in get_ngrams(text, ngram_size):
        m.update(gram.encode("utf-8"))
    return m


def deduplicate(
    df: pd.DataFrame,
    num_perm: int = 128,
    threshold: float = 0.8,
    ngram_size: int = 5,
) -> pd.DataFrame:
    """Remove near-duplicate texts using MinHash LSH.

    Args:
        df: DataFrame with 'text' column.
        num_perm: Number of permutations for MinHash.
        threshold: Jaccard similarity threshold for considering duplicates.
        ngram_size: Size of character n-grams for shingling.

    Returns:
        Deduplicated DataFrame.
    """
    original_size = len(df)
    logger.info(f"Starting deduplication on {original_size} records (threshold={threshold})...")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep_indices = []

    for idx, row in df.iterrows():
        mh = build_minhash(row["text"], num_perm=num_perm, ngram_size=ngram_size)

        # Check if this document is a near-duplicate of any already inserted
        try:
            result = lsh.query(mh)
        except ValueError:
            result = []

        if not result:
            # No near-duplicate found, keep this document
            key = f"doc_{idx}"
            try:
                lsh.insert(key, mh)
                keep_indices.append(idx)
            except ValueError:
                # Duplicate MinHash insertion, skip
                pass

    df_dedup = df.loc[keep_indices].reset_index(drop=True)
    removed = original_size - len(df_dedup)
    logger.info(f"Deduplication complete. Removed {removed} duplicates ({removed/original_size*100:.1f}%). "
                f"{len(df_dedup)} records remaining.")
    return df_dedup
