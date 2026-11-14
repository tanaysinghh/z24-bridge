import json
import time
from functools import lru_cache

import numpy as np
import onnxruntime as ort

from vibra import rul
from vibra.config import ARTIFACTS_DIR
from vibra.loaders import InputError
from vibra.minirocket_numpy import transform as minirocket_transform
from vibra.minirocket_numpy import transform_multi as minirocket_transform_multi
from vibra.preprocess import channel_labels, plot_series, prepare, spectrum
from vibra.taxonomy import CLASS_NAMES, CLASS_TO_GROUP, DSI, GROUP_DESCRIPTIONS, GROUP_MEMBERS, GROUP_NAMES

DISCLAIMER = (
    "The remaining useful life is a model-based prognostic estimate derived from the defined damage-severity ordering. "
    "Z24 has no ground-truth lifespan data, so it has not been validated against real remaining-life measurements."
)


def log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def softmax(x, axis=-1):
    return np.exp(log_softmax(x, axis))


def recording(window_logp):
    return softmax(window_logp.mean(axis=0))


class OnnxModel:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def scores(self, x):
        return self.session.run(None, {self.input: x.astype(np.float32)})[0].astype(np.float64)


class MiniRocketModel:
    def __init__(self, path):
        with np.load(path) as f:
            self.p = {k: f[k] for k in f.files}

    def scores(self, x):
        if "channel_indices" in self.p:
            feats = minirocket_transform_multi(
                x, self.p["num_channels_per_combination"], self.p["channel_indices"], self.p["dilations"], self.p["num_features_per_dilation"], self.p["biases"]
            )
        else:
            feats = minirocket_transform(x[:, 0, :], self.p["dilations"], self.p["num_features_per_dilation"], self.p["biases"])
        feats = (feats - self.p["scaler_mean"]) / self.p["scaler_scale"]
        return feats.astype(np.float64) @ self.p["coef"].T + self.p["intercept"]


def load_model(spec):
    path = ARTIFACTS_DIR / spec["file"]
    return MiniRocketModel(path) if spec["type"] == "minirocket" else OnnxModel(path)


class Engine:
    def __init__(self):
        self.manifest = json.loads((ARTIFACTS_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.cfg = self.manifest["preprocessing"]
        self.models = {k: load_model(v) for k, v in self.manifest["models"].items()}
        h = self.manifest["hierarchy"]
        self.stage1 = load_model(h["stage1"])
        self.specialists = {g: load_model(s) for g, s in h["specialists"].items()}
        ens = self.manifest["ensemble"]
        with np.load(ARTIFACTS_DIR / ens["file"]) as f:
            self.meta_coef = f["coef"].astype(np.float64)
            self.meta_intercept = f["intercept"].astype(np.float64)
        with np.load(ARTIFACTS_DIR / self.manifest["samples"]["file"]) as f:
            self.samples = {k: f[k] for k in f.files}
        self.sample_index = json.loads((ARTIFACTS_DIR / self.manifest["samples"]["index"]).read_text(encoding="utf-8"))
        specs = list(self.manifest["models"].values()) + [h["stage1"]] + list(h["specialists"].values())
        self.required_channels = sorted({c for spec in specs for c in spec["channels"]})

    def select(self, windows, labels, channels):
        missing = [c for c in channels if c not in labels]
        if missing:
            raise InputError(f"This model needs Z24 channels {channels}; the input is missing {missing}.", "bad_channels")
        return windows[:, [labels.index(c) for c in channels], :]

    def base_logps(self, windows, labels):
        out = {}
        for key, spec in self.manifest["models"].items():
            scores = self.models[key].scores(self.select(windows, labels, spec["channels"]))
            out[key] = log_softmax(scores / spec["temperature"])
        return out

    def ensemble_logp(self, logps):
        ens = self.manifest["ensemble"]
        order = ens["order"]
        if ens["method"] == "stacking":
            feats = np.concatenate([logps[k] for k in order], axis=1)
            return log_softmax(feats @ self.meta_coef.T + self.meta_intercept)
        weights = np.array([ens["weights"][k] for k in order], dtype=float)
        weights = weights / weights.sum()
        stacked = np.stack([logps[k] + np.log(w) for k, w in zip(order, weights)])
        m = stacked.max(axis=0)
        return m + np.log(np.exp(stacked - m).sum(axis=0))

    def hierarchy(self, windows, labels):
        h = self.manifest["hierarchy"]
        s1 = log_softmax(self.stage1.scores(self.select(windows, labels, h["stage1"]["channels"])))
        group_probs = recording(s1)
        g = GROUP_NAMES[int(group_probs.argmax())]
        spec = h["specialists"][g]
        s2 = log_softmax(self.specialists[g].scores(self.select(windows, labels, spec["channels"])))
        within = recording(s2)
        members = GROUP_MEMBERS[g]
        best = int(within.argmax())
        return {
            "group": {
                "name": g,
                "probability": float(group_probs.max()),
                "description": GROUP_DESCRIPTIONS[g],
                "probabilities": {GROUP_NAMES[i]: float(p) for i, p in enumerate(group_probs)},
            },
            "scenario": {
                "label": members[best],
                "name": CLASS_NAMES[members[best]],
                "probability_within_group": float(within[best]),
                "within_group": [{"label": c, "name": CLASS_NAMES[c], "probability": float(within[i])} for i, c in enumerate(members)],
            },
        }

    def predict(self, signal, fs, labels=None, already_filtered=False, truth=None, source=None, warnings=None):
        t0 = time.perf_counter()
        signal = np.atleast_2d(np.asarray(signal, dtype=np.float64))
        labels = labels or channel_labels(signal.shape[0])
        missing = [c for c in self.required_channels if c not in labels]
        if missing:
            raise InputError(
                f"The deployed models use Z24 reference sensors {self.required_channels[0]}-{self.required_channels[-1]}, but this file has "
                f"{signal.shape[0]} channel{'s' if signal.shape[0] != 1 else ''}. Upload {len(self.required_channels)} channels "
                f"({', '.join(str(c) for c in self.required_channels)} in that order) or all 27 Z24 channels.",
                "bad_channels",
            )
        filtered, windows, starts = prepare(signal, fs, self.cfg, already_filtered)
        logps = self.base_logps(windows, labels)
        ens_w = self.ensemble_logp(logps)
        probs = recording(ens_w)
        top = int(probs.argmax())
        models = []
        for key, spec in self.manifest["models"].items():
            p = recording(logps[key])
            k = int(p.argmax())
            models.append(
                {
                    "key": key,
                    "name": spec["name"],
                    "label": k,
                    "scenario": CLASS_NAMES[k],
                    "group": CLASS_TO_GROUP[k],
                    "probability": float(p[k]),
                    "probability_of_ensemble_class": float(p[top]),
                    "agrees": k == top,
                    "agrees_group": CLASS_TO_GROUP[k] == CLASS_TO_GROUP[top],
                    "probabilities": [float(v) for v in p],
                }
            )
        hier = self.hierarchy(windows, labels)
        est = rul.estimate(probs, seed=0)
        win_pred = ens_w.argmax(1)
        primary_row = labels.index(22) if 22 in labels else 0
        trace = filtered[primary_row]
        result = {
            "input": {
                "source": source,
                "n_channels": int(signal.shape[0]),
                "channels": labels,
                "fs_input": float(fs),
                "fs_model": self.cfg["fs"],
                "duration_s": float(signal.shape[-1] / fs),
                "n_windows": int(len(windows)),
                "warnings": warnings or [],
            },
            "prediction": {
                "label": top,
                "scenario": CLASS_NAMES[top],
                "group": CLASS_TO_GROUP[top],
                "probability": float(probs[top]),
                "method": self.manifest["ensemble"]["label"],
                "probabilities": [{"label": i, "name": CLASS_NAMES[i], "group": CLASS_TO_GROUP[i], "probability": float(p)} for i, p in enumerate(probs)],
            },
            "agreement": {
                "models_agreeing": int(sum(m["agrees"] for m in models)),
                "models_total": len(models),
                "hierarchy_agrees": hier["scenario"]["label"] == top,
                "hierarchy_group_agrees": hier["group"]["name"] == CLASS_TO_GROUP[top],
                "models": models,
            },
            "hierarchy": hier,
            "severity": {
                "dsi": est["dsi"],
                "defined_dsi_of_prediction": DSI[top],
                "tier": est["tier"],
                "tier_probabilities": est["tier_probabilities"],
            },
            "rul": {**est["rul_years"], "interval": est["interval"], "histogram": est["histogram"], "params": est["params"], "disclaimer": DISCLAIMER},
            "windows": [
                {"start_s": float(s / self.cfg["fs"]), "label": int(c), "scenario": CLASS_NAMES[int(c)], "probability": float(np.exp(ens_w[i, c]))}
                for i, (s, c) in enumerate(zip(starts, win_pred))
            ],
            "signal": plot_series(trace, self.cfg["fs"]),
            "spectrum": spectrum(trace, self.cfg["fs"]),
            "elapsed_ms": None,
        }
        if truth is not None:
            result["truth"] = truth
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def predict_sample(self, sample_id):
        index = {s["id"]: s for s in self.sample_index["samples"]}
        if sample_id not in index:
            raise KeyError(sample_id)
        s = index[sample_id]
        x = self.samples["X"][s["row"]].astype(np.float64)
        truth = {"label": s["label"], "scenario": s["scenario"], "group": s["group"], "setup": s["setup"], "defined_dsi": DSI[s["label"]]}
        return self.predict(x, self.cfg["fs"], self.sample_index["channels"], already_filtered=True, truth=truth, source=f"Built-in sample {sample_id}")


@lru_cache(maxsize=1)
def get_engine():
    return Engine()
