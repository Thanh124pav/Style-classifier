"""Main training pipeline for Vietnamese text style classification."""

import argparse
import json
import logging
import os

import yaml

from src.data_loader import load_data, split_data
from src.dataset import StyleDataset
from src.dedup import deduplicate
from src.model import load_model_and_tokenizer
from src.preprocessing import preprocess_dataframe
from src.trainer import evaluate_model, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    labels = config["labels"]
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    # 1. Load data
    logger.info("=== Loading data ===")
    df = load_data(
        data_cfg["input_path"],
        text_field=data_cfg.get("text_field", "text"),
        label_field=data_cfg.get("label_field", "label"),
    )

    # Validate labels
    unknown = set(df["label"].unique()) - set(labels)
    if unknown:
        logger.warning(f"Unknown labels found and will be dropped: {unknown}")
        df = df[df["label"].isin(labels)].reset_index(drop=True)

    # 2. Preprocess
    logger.info("=== Preprocessing ===")
    df = preprocess_dataframe(df, config)

    # 3. Deduplicate
    dedup_cfg = config.get("dedup", {})
    if dedup_cfg.get("enabled", True):
        logger.info("=== Deduplication ===")
        df = deduplicate(
            df,
            num_perm=dedup_cfg.get("num_perm", 128),
            threshold=dedup_cfg.get("threshold", 0.8),
            ngram_size=dedup_cfg.get("ngram_size", 5),
        )

    # 4. Split data
    logger.info("=== Splitting data ===")
    train_df, val_df, test_df = split_data(
        df,
        test_size=data_cfg.get("test_size", 0.15),
        val_size=data_cfg.get("val_size", 0.1),
        seed=data_cfg.get("seed", 42),
    )

    # Save processed data
    output_path = data_cfg.get("output_path", "data/processed")
    os.makedirs(output_path, exist_ok=True)
    train_df.to_json(os.path.join(output_path, "train.jsonl"), orient="records", lines=True, force_ascii=False)
    val_df.to_json(os.path.join(output_path, "val.jsonl"), orient="records", lines=True, force_ascii=False)
    test_df.to_json(os.path.join(output_path, "test.jsonl"), orient="records", lines=True, force_ascii=False)
    logger.info(f"Processed data saved to {output_path}")

    # 5. Load model and tokenizer
    logger.info("=== Loading model ===")
    model_cfg = config["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        num_labels=model_cfg.get("num_labels", len(labels)),
        label2id=label2id,
        id2label=id2label,
    )

    # 6. Create datasets
    logger.info("=== Creating datasets ===")
    max_length = model_cfg.get("max_length", 256)

    train_labels = [label2id[l] for l in train_df["label"]]
    val_labels = [label2id[l] for l in val_df["label"]]
    test_labels = [label2id[l] for l in test_df["label"]]

    train_dataset = StyleDataset(train_df["text"].tolist(), train_labels, tokenizer, max_length)
    val_dataset = StyleDataset(val_df["text"].tolist(), val_labels, tokenizer, max_length)
    test_dataset = StyleDataset(test_df["text"].tolist(), test_labels, tokenizer, max_length)

    # 7. Train
    logger.info("=== Training ===")
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, config)

    # 8. Evaluate
    logger.info("=== Evaluation ===")
    label_display = config.get("label_display", {})
    display_names = [label_display.get(l, l) for l in labels]
    metrics = evaluate_model(trainer, test_dataset, label_names=display_names)

    # Save metrics
    train_cfg = config.get("training", {})
    output_dir = train_cfg.get("output_dir", "outputs")
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Test metrics saved to {metrics_path}")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vietnamese text style classifier")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    main(args.config)
