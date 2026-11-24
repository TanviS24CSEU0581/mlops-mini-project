import json
import logging
import sys
from pathlib import Path

# Ensure src/ is in sys.path so mlpkg can be imported from anywhere
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlpkg.cleaning import normalise_tier

# -------------------------------------------------------------------
# 1. Structured JSON Logging (Student Handbook Section 9 & Section 11-J)
# -------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """Outputs log records as single-line JSON to stdout."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            log_payload.update(record.extra_fields)
        return json.dumps(log_payload)


logger = logging.getLogger("mp114_api")
logger.setLevel(logging.INFO)
logger.propagate = False  # Avoid duplicate root logs

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(JsonFormatter())
logger.handlers = [stream_handler]


# -------------------------------------------------------------------
# 2. Pydantic Request & Response Schemas (PROBLEM.md Section 5)
# -------------------------------------------------------------------
class PredictRequest(BaseModel):
    age: float = Field(..., description="Customer age in years")
    screen_hours: float = Field(..., description="Screen time in hours")
    # Rule 1: apps_installed above 350 is impossible -> reject with 422
    apps_installed: int = Field(..., le=350, description="Apps installed (must be <= 350)")
    # Rule 2: plan_gb_limit below 0 is always a typing mistake -> reject with 422
    plan_gb_limit: float = Field(..., ge=0.0, description="Daily data plan limit in GB (must be >= 0)")
    video_minutes: float = Field(..., description="Video watched in minutes")
    wifi_hours: float = Field(..., description="Time on home Wi-Fi in hours")
    phone_tier: str = Field(..., description="Phone tier: basic, mid, or flagship")
    circle: str = Field(..., description="Area type: Urban, Semi-urban, or Rural")


class PredictResponse(BaseModel):
    prediction: float


# -------------------------------------------------------------------
# 3. FastAPI Application
# -------------------------------------------------------------------
app = FastAPI(title="MP114 Mobile Data Predictor", version="1.0")

MODEL_PATH = BASE_DIR / "models" / "model.joblib"
model = None


def get_model():
    """Lazy loader for model artifact."""
    global model
    if model is None:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
        else:
            raise RuntimeError(f"Champion model not found at {MODEL_PATH}")
    return model


@app.get("/health")
def health():
    """GET /health must return status ok."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    POST /predict takes the exact 8 fields, applies pydantic rules,
    normalises tier, generates the prediction, and writes one JSON log line.
    """
    try:
        active_model = get_model()
    except Exception as exc:
        logger.error(f"Model unavailable: {exc}")
        raise HTTPException(status_code=503, detail="Model unavailable")

    # Build single-row DataFrame
    row_dict = payload.model_dump()
    df = pd.DataFrame([row_dict])
    df["phone_tier"] = normalise_tier(df["phone_tier"])

    try:
        prediction_val = float(active_model.predict(df)[0])
    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(status_code=500, detail="Inference calculation failed")

    # Structured JSON log: exactly one line to stdout, no personal info
    logger.info(
        "Prediction generated",
        extra={
            "extra_fields": {
                "prediction": round(prediction_val, 4),
                "circle": payload.circle,
            }
        },
    )

    return PredictResponse(prediction=prediction_val)
