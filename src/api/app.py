import os
from typing import Optional

from fastapi import FastAPI, HTTPException

from src.api.schemas import InputData
from src.models.inference import MorphologyPredictor, load_predictor
from src.utils.logging import configure_logging, get_logger
from src.utils.paths import resolve_path


LOGGER = get_logger(__name__)
configure_logging()

DEFAULT_MODEL_DIR = str(resolve_path("trained_arabart_morph_model"))
MODEL_DIR = os.getenv("ARABART_MODEL_DIR", DEFAULT_MODEL_DIR)

app = FastAPI(title="Arabic Morphology API", version="1.0.0")
_predictor: Optional[MorphologyPredictor] = None


def get_predictor() -> MorphologyPredictor:
    global _predictor
    if _predictor is None:
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(
                f"Model directory does not exist: {MODEL_DIR}. "
                "Train a model or set ARABART_MODEL_DIR to a valid path."
            )
        _predictor = load_predictor(MODEL_DIR)
    return _predictor


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_dir": MODEL_DIR}


@app.post("/analyze")
def analyze_text(data: InputData) -> dict:
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Failed to initialize predictor.")
        raise HTTPException(status_code=500, detail="Model initialization failed.") from exc

    return {"result": predictor.analyze_text(data.text)}

