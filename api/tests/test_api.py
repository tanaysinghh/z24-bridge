import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import savemat

from main import app
from vibra.config import ARTIFACTS_DIR, MAX_UPLOAD_BYTES

client = TestClient(app)
MANIFEST = json.loads((ARTIFACTS_DIR / "manifest.json").read_text())
NEEDS_FIVE = any(len(m["channels"]) > 1 for m in MANIFEST["models"].values())
ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(0)


def sample_signal(seconds=60, channels=1, fs=100.0):
    t = np.arange(int(seconds * fs)) / fs
    base = np.sin(2 * np.pi * 3.9 * t) + 0.5 * np.sin(2 * np.pi * 5.1 * t)
    return np.stack([base + 0.3 * RNG.standard_normal(len(t)) for _ in range(channels)])


def npy_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def post(name, raw, fs=None):
    data = {"fs": str(fs)} if fs is not None else {}
    return client.post("/api/predict", files={"file": (name, raw, "application/octet-stream")}, data=data)


def check_prediction(body):
    probs = [c["probability"] for c in body["prediction"]["probabilities"]]
    assert len(probs) == 17
    assert abs(sum(probs) - 1) < 1e-6
    assert body["prediction"]["label"] == int(np.argmax(probs))
    assert 0 <= body["severity"]["dsi"]["expected"] <= 1
    assert body["rul"]["lower"] <= body["rul"]["median"] <= body["rul"]["upper"]
    assert body["rul"]["interval"] == 0.9
    assert "not been validated" in body["rul"]["disclaimer"]
    assert body["hierarchy"]["group"]["name"] in {"Healthy", "Settlement", "Foundation", "Hinge", "Tendon"}
    assert body["hierarchy"]["scenario"]["label"] in [s["label"] for s in body["hierarchy"]["scenario"]["within_group"]]
    assert len(body["agreement"]["models"]) == 3
    assert body["severity"]["tier"]["action"]
    assert len(body["signal"]["t"]) == len(body["signal"]["y"]) > 0
    assert len(body["spectrum"]["f"]) == len(body["spectrum"]["db"]) > 0


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == 9


def test_meta():
    body = client.get("/api/meta").json()
    assert len(body["classes"]) == 17
    assert [g["name"] for g in body["groups"]] == ["Healthy", "Settlement", "Foundation", "Hinge", "Tendon"]
    assert sorted(c for g in body["groups"] for c in g["members"]) == list(range(17))
    assert len(body["tiers"]) == 5


def test_samples_and_sample_prediction():
    index = client.get("/api/samples").json()
    assert len(index["samples"]) == 34
    assert {s["setup"] for s in index["samples"]} == {8, 9}
    first = index["samples"][0]
    r = client.get(f"/api/samples/{first['id']}/prediction")
    assert r.status_code == 200
    body = r.json()
    check_prediction(body)
    assert body["truth"]["label"] == first["label"]
    assert body["input"]["n_windows"] == 5


def test_sample_predictions_match_evaluation_pipeline():
    index = client.get("/api/samples").json()
    if index["note"] == "DEV":
        pytest.skip("development artifacts")
    for s in index["samples"]:
        body = client.get(f"/api/samples/{s['id']}/prediction").json()
        assert body["prediction"]["label"] == s["expected"]["label"]
        assert abs(body["prediction"]["probability"] - s["expected"]["probability"]) < 0.02


def test_unknown_sample():
    assert client.get("/api/samples/nope/prediction").status_code == 404


def test_upload_npy_single_channel():
    r = post("rec.npy", npy_bytes(sample_signal()[0].astype(np.float32)))
    if NEEDS_FIVE:
        assert r.status_code == 422
        assert r.json()["code"] == "bad_channels"
        assert "22" in r.json()["detail"]
        return
    assert r.status_code == 200, r.text
    body = r.json()
    check_prediction(body)
    assert body["input"]["channels"] == [22]


def test_upload_assumes_100hz_without_rate():
    r = post("rec.npy", npy_bytes(sample_signal(channels=5).astype(np.float32)))
    assert r.status_code == 200, r.text
    check_prediction(r.json())
    assert any("assuming 100 Hz" in w for w in r.json()["input"]["warnings"])


def test_upload_27_channels():
    r = post("rec.npy", npy_bytes(sample_signal(seconds=30, channels=27)), fs=100)
    assert r.status_code == 200, r.text
    assert r.json()["input"]["n_channels"] == 27


def test_upload_npy_five_channels_columns():
    r = post("rec.npy", npy_bytes(sample_signal(channels=5).T), fs=100)
    assert r.status_code == 200, r.text
    assert r.json()["input"]["channels"] == [22, 23, 24, 25, 26]


def test_upload_csv_with_time_column_detects_rate():
    x = sample_signal(seconds=40, fs=200)[0]
    t = np.arange(len(x)) / 200
    text = "time_s,acc\n" + "\n".join(f"{a:.4f},{b:.6f}" for a, b in zip(t, x))
    r = post("rec.csv", text.encode())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input"]["fs_input"] == pytest.approx(200, rel=1e-3)
    assert body["input"]["n_windows"] == 3
    assert any("Resampled" in w for w in body["input"]["warnings"])


def test_upload_csv_without_header():
    text = "\n".join(f"{v:.6f}" for v in sample_signal(seconds=25)[0])
    r = post("rec.csv", text.encode(), fs=100)
    assert r.status_code == 200, r.text


def test_upload_mat():
    buf = io.BytesIO()
    savemat(buf, {"data": sample_signal(channels=5)})
    r = post("rec.mat", buf.getvalue(), fs=100)
    assert r.status_code == 200, r.text
    assert r.json()["input"]["n_channels"] == 5


def test_public_example_files_run():
    examples = json.loads((ROOT / "web" / "public" / "samples" / "index.json").read_text())
    for ex in examples:
        raw = (ROOT / "web" / "public" / ex["csv"]).read_bytes()
        r = post(Path(ex["csv"]).name, raw)
        assert r.status_code == 200, r.text
        assert r.json()["input"]["fs_input"] == pytest.approx(100)


@pytest.mark.parametrize(
    "name,raw,fs,code",
    [
        ("rec.wav", b"RIFF0000", None, "unsupported_type"),
        ("rec.npy", b"", None, "empty"),
        ("rec.npy", b"not a numpy file", None, "unreadable"),
        ("rec.csv", b"a,b\nx,y\n", None, "non_numeric"),
        ("rec.npy", npy_bytes(sample_signal(seconds=5, channels=5)), None, "too_short"),
        ("rec.npy", npy_bytes(np.zeros((5, 3000))), None, "flat_signal"),
        ("rec.npy", npy_bytes(np.full(3000, np.nan)), None, "non_finite"),
        ("rec.npy", npy_bytes(sample_signal(channels=3)), None, "bad_channels"),
        ("rec.npy", npy_bytes(np.zeros((2, 3, 4))), None, "bad_shape"),
        ("rec.npy", npy_bytes(sample_signal(channels=5)), 10, "bad_fs"),
        ("rec.npy", npy_bytes(sample_signal(seconds=700, channels=5)), None, "too_long"),
    ],
)
def test_upload_errors(name, raw, fs, code):
    r = post(name, raw, fs)
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == code
    assert len(body["detail"]) > 10


def test_upload_too_large():
    r = post("big.npy", b"0" * (MAX_UPLOAD_BYTES + 10))
    assert r.status_code == 422
    assert r.json()["code"] == "too_large"


def test_missing_file_field():
    assert client.post("/api/predict").status_code == 422


def test_cors_allows_frontend_origin():
    for origin in ("https://vibradiagnose.vercel.app", "http://localhost:5173"):
        r = client.options("/api/predict", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
        assert r.headers.get("access-control-allow-origin") == origin


def test_cors_blocks_other_origin():
    r = client.options("/api/predict", headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


def test_metrics_endpoint():
    assert client.get("/api/metrics").status_code == 200


def test_shared_modules_in_sync_with_src():
    pairs = {"taxonomy.py": ROOT / "src" / "taxonomy.py", "rul.py": ROOT / "src" / "rul.py", "minirocket_numpy.py": ROOT / "src" / "inference" / "minirocket_numpy.py"}
    if not all(p.exists() for p in pairs.values()):
        pytest.skip("training sources not present")
    for name, src in pairs.items():
        expected = re.sub(r"\bfrom src\.", "from vibra.", src.read_text(encoding="utf-8"))
        assert (Path(ARTIFACTS_DIR).parent / "vibra" / name).read_text(encoding="utf-8") == expected, name
