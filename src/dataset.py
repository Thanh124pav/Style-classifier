"""PyTorch Dataset for text classification with chunking support."""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class StyleDataset(Dataset):
    """Dataset for style classification that handles long texts via truncation.

    For training, texts are truncated to max_length.
    For inference with chunking, use the chunking module separately.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
