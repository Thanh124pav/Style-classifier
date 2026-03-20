"""Model module for encoder-only text classification."""

import logging

from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int = 5,
    label2id: dict | None = None,
    id2label: dict | None = None,
):
    """Load a pretrained encoder-only model and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model identifier (e.g., 'vinai/phobert-base-v2').
        num_labels: Number of classification labels.
        label2id: Mapping from label name to index.
        id2label: Mapping from index to label name.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure tokenizer has a pad token (some models like GPT-2 lack one)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Ensure tokenizer has an EOS token for sequence classification
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is not None:
            tokenizer.eos_token = tokenizer.sep_token
        else:
            tokenizer.add_special_tokens({"eos_token": "</s>"})

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    # Resize embeddings if new special tokens were added
    model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer
