"""FastAPI server for Vietnamese text style classification.

Usage:
    python serve.py --model_path outputs/best_model
    python serve.py --model_path outputs/best_model --port 8000 --workers 1

API:
    POST /predict         → classify single text
    POST /predict_batch   → classify multiple texts
    GET  /health          → health check
    GET  /labels          → list available labels
"""

import asyncio
import argparse
import logging
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import StyleClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Vietnamese Style Classifier", version="1.0")

# Global classifier instance (loaded at startup)
classifier: StyleClassifier | None = None
# Lock to serialize GPU inference (GPU can't truly parallelize)
# Created lazily on first request
_inference_lock: asyncio.Lock | None = None

def get_lock() -> asyncio.Lock:
    global _inference_lock
    if _inference_lock is None:
        _inference_lock = asyncio.Lock()
    return _inference_lock


# ── Request/Response models ───────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    return_probs: bool = False

class PredictBatchRequest(BaseModel):
    texts: list[str]
    return_probs: bool = False

class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities: dict[str, float] | None = None
    latency_ms: float

class BatchResponse(BaseModel):
    results: list[PredictResponse]
    total_latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "device": str(classifier.device) if classifier else None,
    }


@app.get("/labels")
async def labels():
    if classifier is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "labels": classifier.id2label,
        "num_labels": len(classifier.id2label),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if classifier is None:
        raise HTTPException(503, "Model not loaded")
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    start = time.perf_counter()
    # Run GPU inference in thread pool to not block the event loop,
    # lock ensures GPU calls don't overlap (GPU can't truly parallelize)
    loop = asyncio.get_event_loop()
    async with get_lock():
        result = await loop.run_in_executor(
            None, classifier.predict, req.text, req.return_probs
        )
    latency = (time.perf_counter() - start) * 1000

    return PredictResponse(
        label=result["label"],
        label_id=result["label_id"],
        confidence=result["confidence"],
        probabilities=result.get("probabilities"),
        latency_ms=round(latency, 2),
    )


@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(req: PredictBatchRequest):
    if classifier is None:
        raise HTTPException(503, "Model not loaded")
    if not req.texts:
        raise HTTPException(400, "Texts list cannot be empty")

    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    results = []
    async with get_lock():
        for text in req.texts:
            t0 = time.perf_counter()
            result = await loop.run_in_executor(
                None, classifier.predict, text, req.return_probs
            )
            t1 = time.perf_counter()
            results.append(PredictResponse(
                label=result["label"],
                label_id=result["label_id"],
                confidence=result["confidence"],
                probabilities=result.get("probabilities"),
                latency_ms=round((t1 - t0) * 1000, 2),
            ))
    total = (time.perf_counter() - start) * 1000

    return BatchResponse(results=results, total_latency_ms=round(total, 2))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global classifier

    parser = argparse.ArgumentParser(description="Serve style classifier API")
    parser.add_argument("--model_path", required=True, help="Path to trained model directory")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu, or auto (default)")
    parser.add_argument("--no-word-segment", action="store_true", help="Disable word segmentation")
    args = parser.parse_args()

    logger.info("Loading model from %s ...", args.model_path)
    classifier = StyleClassifier(
        model_path=args.model_path,
        max_length=args.max_length,
        device=args.device,
        do_word_segment=not args.no_word_segment,
    )

    logger.info("Model loaded. Labels: %s", list(classifier.id2label.values()))
    logger.info("Starting server on %s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
