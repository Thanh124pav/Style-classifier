"""Inference script for Vietnamese text style classification."""

import argparse
import json
import logging
import sys

import yaml

from src.inference import StyleClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def predict_interactive(classifier: StyleClassifier):
    """Interactive prediction mode - read texts from stdin."""
    print("Vietnamese Text Style Classifier (interactive mode)")
    print("Enter text to classify. Press Ctrl+D (or Ctrl+Z on Windows) to exit.")
    print("-" * 60)

    while True:
        try:
            text = input("\nText: ").strip()
            if not text:
                continue
            result = classifier.predict(text, return_probs=True)
            print(f"  Style: {result['label']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            if "probabilities" in result:
                print("  Probabilities:")
                for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
                    print(f"    {label}: {prob:.4f}")
        except EOFError:
            print("\nExiting.")
            break


def predict_file(classifier: StyleClassifier, input_path: str, output_path: str):
    """Predict styles for texts in a JSONL file."""
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if not text:
                continue
            pred = classifier.predict(text, return_probs=True)
            record["predicted_label"] = pred["label"]
            record["confidence"] = pred["confidence"]
            record["probabilities"] = pred["probabilities"]
            results.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Predictions saved to {output_path} ({len(results)} records)")


def main():
    parser = argparse.ArgumentParser(description="Predict Vietnamese text styles")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input JSONL file. If not provided, runs in interactive mode.",
    )
    parser.add_argument(
        "--output", type=str, default="predictions.jsonl",
        help="Path to output JSONL file (used with --input)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    chunking_cfg = config.get("chunking", {})
    model_cfg = config.get("model", {})
    preproc_cfg = config.get("preprocessing", {})

    classifier = StyleClassifier(
        model_path=args.model_path,
        max_length=model_cfg.get("max_length", 256),
        chunk_max_length=chunking_cfg.get("max_length", 512),
        chunk_stride=chunking_cfg.get("stride", 128),
        aggregation=chunking_cfg.get("aggregation", "mean"),
        do_word_segment=preproc_cfg.get("word_segment", True),
    )

    if args.input:
        predict_file(classifier, args.input, args.output)
    else:
        predict_interactive(classifier)


if __name__ == "__main__":
    main()
