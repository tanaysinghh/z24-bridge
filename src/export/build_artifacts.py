import argparse
import hashlib
import json
import pickle
import re
import shutil
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import torch

from src.models.inceptiontime import InceptionTime
from src.models.wavenet import WaveNetClassifier
from src.preprocessing.pipeline import PreprocessConfig, load_processed
from src.taxonomy import CLASS_NAMES, CLASS_TO_GROUP, GROUP_MEMBERS, GROUP_NAMES
from src.train.common import MODELS_DIR, RESULTS_DIR

ROOT = Path(__file__).resolve().parents[2]
API = ROOT / "api"
ART = API / "artifacts"
WEB_SAMPLES = ROOT / "web" / "public" / "samples"
SHARED = {"taxonomy.py": ROOT / "src" / "taxonomy.py", "rul.py": ROOT / "src" / "rul.py", "minirocket_numpy.py": ROOT / "src" / "inference" / "minirocket_numpy.py"}
NAMES = {"wavenet": "WaveNet", "inceptiontime": "InceptionTime", "minirocket": "MiniRocket"}
BUILDERS = {"wavenet": WaveNetClassifier, "inceptiontime": InceptionTime}
METHOD_LABELS = {"stacking": "Stacking ensemble (logistic regression)", "simple_average": "Simple average ensemble", "val_weighted_average": "Validation-weighted average ensemble"}
SAMPLE_CHANNELS = [22, 23, 24, 25, 26]
UPLOAD_EXAMPLES = [0, 4, 11, 16]


def sync_shared():
    for name, src in SHARED.items():
        text = src.read_text(encoding="utf-8")
        text = re.sub(r"\bfrom src\.", "from vibra.", text)
        (API / "vibra" / name).write_text(text, encoding="utf-8")


def data_channels(data_name):
    tag = data_name.split("_")[1][2:]
    if "-" in tag:
        a, b = tag.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(c) for c in tag.split("_")]


def export_torch(run, arch, out_name):
    state = torch.load(MODELS_DIR / run / "best.pt", map_location="cpu", weights_only=False)
    args = state["args"]
    kwargs = {"in_channels": state["in_channels"], "num_classes": state["n_classes"]}
    if arch == "wavenet":
        kwargs["dropout"] = args.get("dropout", 0.1)
    model = BUILDERS[arch](**kwargs)
    model.load_state_dict(state["model_state"])
    model.eval()
    path = ART / out_name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            torch.zeros(1, state["in_channels"], 2000),
            str(path),
            input_names=["x"],
            output_names=["logits"],
            dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
            opset_version=17,
            dynamo=False,
        )
    return {"file": out_name, "type": "onnx", "channels": data_channels(args["data"]), "n_classes": state["n_classes"], "source_run": run}


def export_minirocket(run, out_name):
    with open(MODELS_DIR / run / "model.pkl", "rb") as f:
        clf = pickle.load(f)
    params = clf.transform.parameters
    extra = {}
    if len(params) == 5:
        ncpc, chan_idx, dilations, nfpd, biases = params
        extra = {"num_channels_per_combination": np.asarray(ncpc, dtype=np.int64), "channel_indices": np.asarray(chan_idx, dtype=np.int64)}
    else:
        dilations, nfpd, biases = params
    np.savez_compressed(
        ART / out_name,
        **extra,
        dilations=np.asarray(dilations, dtype=np.int64),
        num_features_per_dilation=np.asarray(nfpd, dtype=np.int64),
        biases=np.asarray(biases, dtype=np.float32),
        scaler_mean=clf.scaler.mean_.astype(np.float32),
        scaler_scale=clf.scaler.scale_.astype(np.float32),
        coef=clf.ridge.coef_.astype(np.float32),
        intercept=clf.ridge.intercept_.astype(np.float32),
    )
    payload = json.loads((RESULTS_DIR / f"{run}.json").read_text())
    return {"file": out_name, "type": "minirocket", "channels": data_channels(payload["data"]), "n_classes": 17, "source_run": run}


def build_samples(ensemble_method, seed):
    data = load_processed("z24_ch22-26_w2000_s1000")
    seg_ids = data["test_segment_id"]
    with np.load(MODELS_DIR / "final" / f"ensemble_preds_s{seed}.npz") as f:
        ens_logp, ens_seg = f[f"test_{ensemble_method}"], f["test_window_segment_id"]
    rows, index = [], []
    for c in range(17):
        for setup in (8, 9):
            sid = c * 90 + (setup - 1) * 10
            row = int(np.flatnonzero(seg_ids == sid)[0])
            mean_logp = ens_logp[ens_seg == sid].mean(axis=0)
            p = np.exp(mean_logp - mean_logp.max())
            p /= p.sum()
            rows.append(data["test_segments"][row])
            index.append(
                {
                    "id": f"c{c:02d}-s{setup}",
                    "row": len(rows) - 1,
                    "label": c,
                    "scenario": CLASS_NAMES[c],
                    "group": CLASS_TO_GROUP[c],
                    "setup": setup,
                    "segment_id": sid,
                    "duration_s": 60.0,
                    "expected": {"label": int(p.argmax()), "scenario": CLASS_NAMES[int(p.argmax())], "probability": float(p.max())},
                }
            )
    np.savez_compressed(ART / "samples.npz", X=np.stack(rows).astype(np.float32))
    return {
        "channels": SAMPLE_CHANNELS,
        "note": "Test-set recordings (setups 8-9, first 60 s segment of each). Signals are band-passed exactly as in the evaluation pipeline. 'expected' is the evaluation pipeline's recording-level ensemble prediction for the same recording.",
        "samples": index,
    }


def build_upload_examples():
    raw = np.load(ROOT / "data" / "z24" / "inputs.npy", mmap_mode="r")
    if WEB_SAMPLES.exists():
        shutil.rmtree(WEB_SAMPLES)
    WEB_SAMPLES.mkdir(parents=True)
    out = []
    for c in UPLOAD_EXAMPLES:
        sid = c * 90 + (8 - 1) * 10 + 3
        x = np.asarray(raw[sid, SAMPLE_CHANNELS, :], dtype=np.float32)
        stem = f"z24_test_setup8_scenario{c:02d}"
        t = np.arange(x.shape[1]) / 100.0
        lines = ["time_s,ch22"] + [f"{ti:.2f},{v:.6g}" for ti, v in zip(t, x[0])]
        (WEB_SAMPLES / f"{stem}_ch22.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        np.save(WEB_SAMPLES / f"{stem}_ch22-26.npy", x)
        out.append({"label": c, "scenario": CLASS_NAMES[c], "csv": f"samples/{stem}_ch22-26.csv", "npy": f"samples/{stem}_ch22-26.npy", "segment_id": sid})
    (WEB_SAMPLES / "index.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def metrics_bundle():
    def load(name):
        p = RESULTS_DIR / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    ens = load("ensemble")
    hier = load("hierarchical")
    rul = load("rul")
    final = load("final_models")
    return {
        "generated": date.today().isoformat(),
        "base_models": final["summary"] if final else None,
        "ensemble": {"primary": ens["primary_method"], "summary": ens["summary"]} if ens else None,
        "hierarchical": {"summary": hier["summary"], "stage2": hier["stage2_summary"]} if hier else None,
        "rul_validation": rul["summary"] if rul else None,
        "source": "docs/results/*.json",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hier-arch", default="wavenet")
    args = parser.parse_args()
    sync_shared()
    if ART.exists():
        shutil.rmtree(ART)
    ART.mkdir(parents=True)
    ens = json.loads((RESULTS_DIR / "ensemble.json").read_text())
    seed = args.seed
    models = {}
    for arch in ("wavenet", "inceptiontime"):
        models[arch] = export_torch(f"final/{arch}_s{seed}", arch, f"{arch}.onnx")
    models["minirocket"] = export_minirocket(f"final/minirocket_s{seed}", "minirocket.npz")
    for k in models:
        models[k]["name"] = NAMES[k]
        models[k]["temperature"] = ens["temperatures"][k]
    shutil.copy(MODELS_DIR / "final" / "stacking_meta.npz", ART / "stacking.npz")
    hierarchy = {"stage1": export_torch(f"hier/{args.hier_arch}_group_s{seed}", args.hier_arch, "hier_stage1.onnx"), "specialists": {}}
    for g in GROUP_NAMES:
        spec = export_torch(f"hier/{args.hier_arch}_{g}_s{seed}", args.hier_arch, f"hier_{g.lower()}.onnx")
        spec["members"] = GROUP_MEMBERS[g]
        hierarchy["specialists"][g] = spec
    primary = ens["primary_method"]
    sample_index = build_samples(primary, seed)
    (ART / "samples.json").write_text(json.dumps(sample_index, indent=2), encoding="utf-8")
    uploads = build_upload_examples()
    cfg = PreprocessConfig()
    manifest = {
        "version": date.today().isoformat(),
        "seed": seed,
        "preprocessing": {"fs": cfg.fs, "band": [cfg.band_low, cfg.band_high], "filter_order": cfg.filter_order, "window": cfg.window, "stride": cfg.stride},
        "models": models,
        "ensemble": {
            "method": primary,
            "label": METHOD_LABELS[primary],
            "order": ["wavenet", "inceptiontime", "minirocket"],
            "file": "stacking.npz",
            "weights": ens["per_seed"][str(seed)]["val_weights"] if primary == "val_weighted_average" else {k: 1.0 for k in models},
        },
        "hierarchy": hierarchy,
        "samples": {"file": "samples.npz", "index": "samples.json"},
        "upload_examples": uploads,
    }
    if primary == "stacking":
        manifest["ensemble"]["weights"] = {k: 1.0 for k in models}
    (ART / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ART / "metrics.json").write_text(json.dumps(metrics_bundle(), indent=2), encoding="utf-8")
    sizes = {p.name: p.stat().st_size for p in ART.iterdir()}
    digest = hashlib.sha256(b"".join((ART / n).read_bytes() for n in sorted(sizes))).hexdigest()[:12]
    print(json.dumps({"files": sizes, "total_mb": round(sum(sizes.values()) / 1e6, 2), "digest": digest}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
