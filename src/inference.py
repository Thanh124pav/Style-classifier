"""Inference module with chunking support for long texts."""

import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .chunking import aggregate_predictions, chunk_text
from .preprocessing import clean_vietnamese_text, word_segment

logger = logging.getLogger(__name__)


class StyleClassifier:
    """Inference wrapper for Vietnamese text style classification."""

    def __init__(
        self,
        model_path: str,
        max_length: int = 256,
        chunk_max_length: int = 512,
        chunk_stride: int = 128,
        aggregation: str = "mean",
        device: str | None = None,
        do_word_segment: bool = True,
    ):
        """Initialize the classifier.

        Args:
            model_path: Path to the saved model directory.
            max_length: Max token length for single-chunk classification.
            chunk_max_length: Max token length per chunk for long texts.
            chunk_stride: Overlap between chunks in tokens.
            aggregation: Chunk aggregation method (mean, max, majority_vote).
            device: Device to use (auto-detected if None).
            do_word_segment: Whether to apply Vietnamese word segmentation.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.max_length = max_length
        self.chunk_max_length = chunk_max_length
        self.chunk_stride = chunk_stride
        self.aggregation = aggregation
        self.do_word_segment = do_word_segment

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        logger.info(f"Classifier loaded from {model_path} on {self.device}")

    def preprocess(self, text: str) -> str:
        """Clean and optionally segment input text."""
        text = clean_vietnamese_text(text)
        if self.do_word_segment:
            text = word_segment(text)
        return text

    @torch.no_grad()
    def predict(self, text: str, return_probs: bool = False) -> dict:
        """Predict the style of a single text, using chunking if necessary.

        Args:
            text: Input text (raw, will be preprocessed).
            return_probs: Whether to return class probabilities.

        Returns:
            Dict with 'label', 'label_id', and optionally 'probabilities'.
        """
        text = self.preprocess(text)

        # Check if text needs chunking
        tokens = self.tokenizer(text, add_special_tokens=False)
        needs_chunking = len(tokens["input_ids"]) > self.max_length - 2

        if needs_chunking:
            chunks = chunk_text(
                text, self.tokenizer, self.chunk_max_length, self.chunk_stride
            )
            chunk_logits = []
            for chunk_enc in chunks:
                inputs = {k: v.to(self.device) for k, v in chunk_enc.items()}
                outputs = self.model(**inputs)
                chunk_logits.append(outputs.logits.squeeze(0))

            logits = aggregate_predictions(chunk_logits, method=self.aggregation)
        else:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in encoding.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)

        probs = torch.softmax(logits.float(), dim=-1)
        pred_id = probs.argmax().item()
        pred_label = self.id2label.get(pred_id, str(pred_id))

        result = {
            "label": pred_label,
            "label_id": pred_id,
            "confidence": probs[pred_id].item(),
        }
        if return_probs:
            result["probabilities"] = {
                self.id2label.get(i, str(i)): p.item()
                for i, p in enumerate(probs)
            }

        return result

    def predict_batch(self, texts: list[str], return_probs: bool = False) -> list[dict]:
        """Predict styles for a batch of texts.

        Args:
            texts: List of input texts.
            return_probs: Whether to return class probabilities.

        Returns:
            List of prediction dicts.
        """
        return [self.predict(text, return_probs=return_probs) for text in texts]
