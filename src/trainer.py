"""Training module using HuggingFace Trainer."""

import logging
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """Compute classification metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def train_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    config: dict,
):
    """Train the model using HuggingFace Trainer.

    Args:
        model: The classification model.
        tokenizer: The tokenizer.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        config: Training configuration dictionary.

    Returns:
        Trained Trainer instance.
    """
    train_cfg = config.get("training", {})
    output_dir = train_cfg.get("output_dir", "outputs")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_epochs", 5),
        per_device_train_batch_size=train_cfg.get("batch_size", 16),
        per_device_eval_batch_size=train_cfg.get("batch_size", 16),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        fp16=train_cfg.get("fp16", True),
        logging_steps=train_cfg.get("logging_steps", 50),
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model=train_cfg.get("metric_for_best_model", "f1_macro"),
        greater_is_better=True,
        report_to="none",
        seed=config.get("data", {}).get("seed", 42),
    )

    callbacks = []
    patience = train_cfg.get("early_stopping_patience", 3)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # Save the best model
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    logger.info(f"Best model saved to {best_model_dir}")

    return trainer


def evaluate_model(trainer, test_dataset, label_names: list[str] | None = None):
    """Evaluate the model on the test set and print classification report.

    Args:
        trainer: Trained Trainer instance.
        test_dataset: Test dataset.
        label_names: List of label names for the report.

    Returns:
        Dict of evaluation metrics.
    """
    logger.info("Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    report = classification_report(
        labels, preds, target_names=label_names, digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")
    print(f"\nClassification Report:\n{report}")

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }
    return metrics
