import json
import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vibra import rul
from vibra.config import ARTIFACTS_DIR, MAX_DURATION_S, MAX_UPLOAD_BYTES, ORIGIN_REGEX, ORIGINS
from vibra.engine import get_engine
from vibra.loaders import SUPPORTED, InputError, parse_upload
from vibra.taxonomy import CLASS_NAMES, CLASS_TO_GROUP, DSI, GROUP_DESCRIPTIONS, GROUP_MEMBERS, GROUP_NAMES, TIERS

STARTED = time.time()

app = FastAPI(
    title="VibraDiagnose API",
    version="1.0.0",
    description="Damage-scenario identification, damage severity and remaining-useful-life estimation for the Z24 bridge from accelerometer recordings.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_origin_regex=ORIGIN_REGEX,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(InputError)
async def input_error_handler(request: Request, exc: InputError):
    return JSONResponse(status_code=422, content={"detail": str(exc), "code": exc.code})


@app.get("/")
def root():
    return {"service": "VibraDiagnose API", "health": "/health", "docs": "/docs", "predict": "/api/predict"}


@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "models_loaded": len(engine.models) + 1 + len(engine.specialists),
        "ensemble": engine.manifest["ensemble"]["label"],
        "artifact_version": engine.manifest["version"],
        "uptime_s": round(time.time() - STARTED, 1),
    }


@app.get("/api/meta")
def meta():
    engine = get_engine()
    m = engine.manifest
    return {
        "classes": [{"label": i, "name": n, "group": CLASS_TO_GROUP[i], "dsi": DSI[i]} for i, n in enumerate(CLASS_NAMES)],
        "groups": [{"name": g, "members": GROUP_MEMBERS[g], "description": GROUP_DESCRIPTIONS[g]} for g in GROUP_NAMES],
        "tiers": TIERS,
        "rul_model": rul.DEFAULTS,
        "preprocessing": m["preprocessing"],
        "models": [{"key": k, "name": v["name"], "channels": v["channels"]} for k, v in m["models"].items()],
        "ensemble": m["ensemble"]["label"],
        "input": {
            "formats": list(SUPPORTED),
            "max_bytes": MAX_UPLOAD_BYTES,
            "max_duration_s": MAX_DURATION_S,
            "min_samples_at_100hz": m["preprocessing"]["window"],
            "channels": "1 channel (Z24 channel 22), 5 channels (22-26) or 27 channels",
        },
    }


@app.get("/api/samples")
def samples():
    return get_engine().sample_index


@app.get("/api/samples/{sample_id}/prediction")
def sample_prediction(sample_id: str):
    try:
        return get_engine().predict_sample(sample_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown sample '{sample_id}'.")


@app.get("/api/metrics")
def metrics():
    return json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), fs: float | None = Form(None)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    signal, info = parse_upload(file.filename, raw)
    warnings = []
    rate = fs
    if rate is None and info["fs_detected"]:
        rate = info["fs_detected"]
        warnings.append(f"Sampling rate {rate:g} Hz detected from the time column.")
    if rate is None:
        rate = 100.0
        warnings.append("No sampling rate given; assuming 100 Hz, the Z24 acquisition rate.")
    if abs(rate - 100.0) > 1e-6:
        warnings.append(f"Resampled from {rate:g} Hz to 100 Hz.")
    return get_engine().predict(signal, rate, source=f"{file.filename} ({info['source']})", warnings=warnings)
