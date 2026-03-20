"""Training pipeline for Vietnamese text style classification.

Default model: bert-base-multilingual-cased (mBERT).
"""

import argparse
import json
import logging
import os

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data_loader import load_data, split_data
from src.dataset import StyleDataset
from src.preprocessing import preprocess_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_MODEL = "bert-base-multilingual-cased"
DEFAULT_MAX_LENGTH = 256
DEFAULT_LABELS = ["bao_chi", "hanh_chinh", "khoa_hoc", "chinh_luan", "sinh_hoat"]


# ── Metrics ───────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# ── Model loading ─────────────────────────────────────────────────────
def load_model_and_tokenizer(model_name, num_labels, label2id, id2label):
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Ensure EOS token exists (needed for some encoder models)
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
    model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{total_params:,}")
    return model, tokenizer


# ── Main ──────────────────────────────────────────────────────────────
def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    labels = config.get("labels", DEFAULT_LABELS)

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("=== Loading data ===")
    df = load_data(
        data_cfg.get("input_path", "data/raw"),
        text_field=data_cfg.get("text_field", "text"),
        label_field=data_cfg.get("label_field", "style"),
    )

    # Drop unknown labels
    unknown = set(df["label"].unique()) - set(labels)
    if unknown:
        logger.warning("Dropping unknown labels: %s", unknown)
        df = df[df["label"].isin(labels)].reset_index(drop=True)

    # ── 2. Preprocess ─────────────────────────────────────────────────
    logger.info("=== Preprocessing ===")
    df = preprocess_dataframe(df, config)

    # ── 3. Split ──────────────────────────────────────────────────────
    logger.info("=== Splitting data ===")
    train_df, val_df, test_df = split_data(
        df,
        test_size=data_cfg.get("test_size", 0.15),
        val_size=data_cfg.get("val_size", 0.1),
        seed=data_cfg.get("seed", 42),
    )

    # Save splits
    output_path = data_cfg.get("output_path", "data/processed")
    os.makedirs(output_path, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.to_json(
            os.path.join(output_path, f"{name}.jsonl"),
            orient="records", lines=True, force_ascii=False,
        )
    logger.info("Saved splits to %s", output_path)

    # ── 4. Load model ─────────────────────────────────────────────────
    logger.info("=== Loading model ===")
    model_name = model_cfg.get("name", DEFAULT_MODEL)
    max_length = model_cfg.get("max_length", DEFAULT_MAX_LENGTH)

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    # ── 5. Create datasets ────────────────────────────────────────────
    logger.info("=== Creating datasets ===")
    train_dataset = StyleDataset(
        train_df["text"].tolist(),
        [label2id[l] for l in train_df["label"]],
        tokenizer, max_length,
    )
    val_dataset = StyleDataset(
        val_df["text"].tolist(),
        [label2id[l] for l in val_df["label"]],
        tokenizer, max_length,
    )
    test_dataset = StyleDataset(
        test_df["text"].tolist(),
        [label2id[l] for l in test_df["label"]],
        tokenizer, max_length,
    )

    # ── 6. Training ───────────────────────────────────────────────────
    logger.info("=== Training ===")
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
        seed=data_cfg.get("seed", 42),
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

    trainer.train()

    # Save best model
    best_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info("Best model saved to %s", best_dir)

    # ── 7. Evaluate ───────────────────────────────────────────────────
    logger.info("=== Evaluation ===")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    test_labels = predictions.label_ids

    label_display = config.get("label_display", {})
    display_names = [label_display.get(l, l) for l in labels]

    report = classification_report(test_labels, preds, target_names=display_names, digits=4)
    logger.info("\nClassification Report:\n%s", report)
    print(f"\nClassification Report:\n{report}")

    metrics = {
        "accuracy": accuracy_score(test_labels, preds),
        "f1_macro": f1_score(test_labels, preds, average="macro"),
        "f1_weighted": f1_score(test_labels, preds, average="weighted"),
    }

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vietnamese text style classifier")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config YAML path")
    args = parser.parse_args()
    main(args.config)
