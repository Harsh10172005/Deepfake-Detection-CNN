"""Deepfake Detection — CNN web app (FastAPI).

Serves a single-page UI and a JSON API:

    GET  /              the web interface
    GET  /api/health    service + model status
    GET  /api/metrics   the model's measured evaluation results
                        (results/metrics.json, produced by evaluate.py —
                        nothing on this page is a hardcoded number)
    POST /api/predict   multipart image upload -> Real/Fake prediction

Configuration via environment variables:
    MODEL_NAME    CNN backbone (default: efficientnet_b0)
    MODEL_WEIGHTS path to the .pth checkpoint
                  (default: checkpoints/best_cnn_efficientnet_b0.pth)
    WEIGHTS_URL   optional https URL to download the checkpoint from at
                  startup when the file is missing (used on hosts where
                  the weights are not part of the deployment, e.g. a
                  GitHub Release asset on Render — see render.yaml)
    PORT          listen port (default: 8080)
"""
import json
import os
import shutil
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from dataset import get_transforms
from inference import load_model, predict_image

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
METRICS_PATH = BASE_DIR / "results" / "metrics.json"
TEMP_DIR = BASE_DIR / "temp_uploads"

MODEL_NAME = os.environ.get("MODEL_NAME", "efficientnet_b0")
MODEL_WEIGHTS = os.environ.get(
    "MODEL_WEIGHTS", str(BASE_DIR / "checkpoints" / "best_cnn_efficientnet_b0.pth")
)
IMG_SIZE = 224
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

WEIGHTS_URL = os.environ.get("WEIGHTS_URL", "")


def ensure_weights() -> None:
    """Download the checkpoint from WEIGHTS_URL if it is missing locally.

    Hosts like Render deploy only the git repository, which deliberately
    does not contain the ~48 MB checkpoint. The file is fetched once at
    startup (atomically, via a .part file) and reused afterwards.
    """
    if os.path.exists(MODEL_WEIGHTS) or not WEIGHTS_URL:
        return
    dest = Path(MODEL_WEIGHTS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"[weights] checkpoint missing; downloading from {WEIGHTS_URL} ...")
    with urllib.request.urlopen(WEIGHTS_URL, timeout=120) as resp, open(part, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1024 * 1024)
    part.replace(dest)
    print(f"[weights] downloaded {dest.stat().st_size} bytes -> {dest}")


# Populated at startup by the lifespan handler.
state = {"device": None, "model": None, "transform": None, "model_error": None}


@asynccontextmanager
async def lifespan(_app):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    state["device"] = device
    _, state["transform"] = get_transforms(img_size=IMG_SIZE)
    try:
        ensure_weights()
        if not os.path.exists(MODEL_WEIGHTS):
            raise FileNotFoundError(f"Weights not found at {MODEL_WEIGHTS}")
        print(f"Loading {MODEL_NAME} on {device} from {MODEL_WEIGHTS} ...")
        state["model"] = load_model(MODEL_NAME, MODEL_WEIGHTS, device)
        print("Model loaded.")
    except Exception as exc:  # keep the site up; /api/predict will report 503
        state["model_error"] = str(exc)
        print(f"WARNING: model failed to load: {exc}")
    yield


app = FastAPI(title="Deepfake Detection API (CNN)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# Evaluation plots (confusion matrix / ROC curve) produced by evaluate.py.
_RESULTS_DIR = BASE_DIR / "results"
_RESULTS_DIR.mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=str(_RESULTS_DIR)), name="results")


def read_metrics():
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


@app.get("/", response_class=HTMLResponse)
async def read_index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if state["model"] is not None else "model_unavailable",
        "model_name": MODEL_NAME,
        "weights": MODEL_WEIGHTS,
        "model_error": state["model_error"],
    }


@app.get("/api/metrics")
async def metrics():
    data = read_metrics()
    if data is None:
        raise HTTPException(status_code=404, detail="No evaluation results found. Run evaluate.py first.")
    return data


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {state['model_error']}")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload an image (PNG, JPG, JPEG, or WebP).",
        )

    TEMP_DIR.mkdir(exist_ok=True)
    temp_path = TEMP_DIR / f"{uuid.uuid4()}{ext}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if temp_path.stat().st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 15 MB).")

        prob = predict_image(str(temp_path), state["model"], state["transform"], state["device"])
        label = "Fake" if prob > 0.5 else "Real"
        confidence = prob if prob > 0.5 else 1.0 - prob

        return {
            "prediction": label,
            "confidence": float(confidence),
            "probability_fake": float(prob),
            "filename": file.filename,
            "model": {"name": MODEL_NAME, "weights": os.path.basename(MODEL_WEIGHTS)},
            "metrics": read_metrics(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
