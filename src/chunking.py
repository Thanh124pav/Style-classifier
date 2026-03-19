"""Chunking module for handling long texts during training and inference."""

import logging
from typing import Optional

import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    stride: int = 128,
) -> list[dict]:
    """Split a long text into overlapping chunks based on token count.

    Args:
        text: Input text string.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum number of tokens per chunk (including special tokens).
        stride: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of tokenized chunk dicts with input_ids, attention_mask, etc.
    """
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    all_ids = encoding["input_ids"]

    # If text fits in one chunk, just tokenize normally
    if len(all_ids) <= max_length - 2:  # Account for [CLS] and [SEP]
        return [
            tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        ]

    # Create overlapping windows
    chunks = []
    step = max_length - 2 - stride  # effective step size
    if step <= 0:
        step = max_length - 2

    for start in range(0, len(all_ids), step):
        chunk_ids = all_ids[start : start + max_length - 2]
        if not chunk_ids:
            break

        # Decode back to text and re-encode with special tokens
        chunk_text_str = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunk_enc = tokenizer(
            chunk_text_str,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        chunks.append(chunk_enc)

        # Stop if we've covered all tokens
        if start + max_length - 2 >= len(all_ids):
            break

    return chunks


def aggregate_predictions(
    logits_list: list[torch.Tensor],
    method: str = "mean",
) -> torch.Tensor:
    """Aggregate predictions from multiple chunks.

    Args:
        logits_list: List of logit tensors, each of shape (num_labels,).
        method: Aggregation method - 'mean', 'max', or 'majority_vote'.

    Returns:
        Aggregated logits tensor of shape (num_labels,).
    """
    if len(logits_list) == 1:
        return logits_list[0]

    stacked = torch.stack(logits_list)  # (num_chunks, num_labels)

    if method == "mean":
        return stacked.mean(dim=0)
    elif method == "max":
        return stacked.max(dim=0).values
    elif method == "majority_vote":
        # Each chunk votes for its predicted class
        votes = stacked.argmax(dim=-1)  # (num_chunks,)
        num_labels = stacked.shape[-1]
        vote_counts = torch.zeros(num_labels, device=stacked.device)
        for v in votes:
            vote_counts[v] += 1
        return vote_counts
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
