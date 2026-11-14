import json

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss

from src.taxonomy import CLASS_NAMES, N_CLASSES, setup_of_segment
from src.train.common import MODELS_DIR, RESULTS_DIR

ARCHS = ["wavenet", "inceptiontime", "minirocket"]
NAMES = {"wavenet": "WaveNet", "inceptiontime": "InceptionTime", "minirocket": "MiniRocket"}
SEEDS = [42, 1, 2]
FOLDS = [1, 2, 3]
C_GRID = np.logspace(-3, 1, 9)


def final_run(arch, seed):
    return f"final/{arch}_s{seed}"


def oof_run(arch, fold):
    return f"oof/{arch}_f{fold}"


def load_split(run, split):
    with np.load(MODELS_DIR / run / "preds.npz") as f:
        key = f"{split}_logp" if f"{split}_logp" in f.files else f"{split}_scores"
        return f[key].astype(np.float64), f[f"{split}_window_segment_id"], f[f"{split}_y"]


def load_oof(arch):
    parts = [load_split(oof_run(arch, k), "oof") for k in FOLDS]
    scores = np.concatenate([p[0] for p in parts])
    seg = np.concatenate([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts])
    order = np.lexsort((np.arange(len(seg)), seg))
    return scores[order], seg[order], y[order]


def fit_temperature(scores, y):
    def nll(log_t):
        lp = log_softmax(scores / np.exp(log_t), axis=1)
        return -lp[np.arange(len(y)), y].mean()

    res = minimize_scalar(nll, bounds=(-6, 6), method="bounded")
    return float(np.exp(res.x))


def calibrated(scores, temperature):
    return log_softmax(scores / temperature, axis=1)


def segment_logp(window_logp, window_segment_id):
    seg_ids, inverse = np.unique(window_segment_id, return_inverse=True)
    summed = np.zeros((len(seg_ids), window_logp.shape[1]))
    np.add.at(summed, inverse, window_logp)
    return seg_ids, summed


def metrics(window_logp, y, window_segment_id, n_classes=N_CLASSES):
    pred = window_logp.argmax(1)
    seg_ids, seg = segment_logp(window_logp, window_segment_id)
    seg_y = np.array([y[window_segment_id == s][0] for s in seg_ids])
    seg_pred = seg.argmax(1)
    cm = confusion_matrix(seg_y, seg_pred, labels=np.arange(n_classes))
    cm_w = confusion_matrix(y, pred, labels=np.arange(n_classes))
    return {
        "window_acc": float((pred == y).mean()),
        "window_macro_f1": float(f1_score(y, pred, average="macro")),
        "segment_acc": float((seg_pred == seg_y).mean()),
        "segment_macro_f1": float(f1_score(seg_y, seg_pred, average="macro")),
        "window_nll": float(-window_logp[np.arange(len(y)), y].mean()),
        "recall_segment": (cm.diagonal() / np.maximum(cm.sum(1), 1)).tolist(),
        "recall_window": (cm_w.diagonal() / np.maximum(cm_w.sum(1), 1)).tolist(),
        "confusion_segment": cm.tolist(),
        "n_windows": int(len(y)),
        "n_segments": int(len(seg_ids)),
    }


def average_logp(logps, weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    stacked = np.stack([lp + np.log(wi) for lp, wi in zip(logps, w)])
    return logsumexp(stacked, axis=0)


def stack_features(logps):
    return np.concatenate(logps, axis=1)


def select_c(features, y, setups):
    scores = {}
    for c in C_GRID:
        losses = []
        for held in np.unique(setups):
            tr, te = setups != held, setups == held
            clf = LogisticRegression(C=float(c), max_iter=3000).fit(features[tr], y[tr])
            losses.append(log_loss(y[te], clf.predict_proba(features[te]), labels=np.arange(N_CLASSES)))
        scores[float(c)] = float(np.mean(losses))
    return min(scores, key=scores.get), scores


def fit_meta():
    oof = {a: load_oof(a) for a in ARCHS}
    seg = oof[ARCHS[0]][1]
    y = oof[ARCHS[0]][2]
    for a in ARCHS[1:]:
        if not (np.array_equal(oof[a][1], seg) and np.array_equal(oof[a][2], y)):
            raise ValueError(f"OOF window order mismatch for {a}")
    temperatures = {a: fit_temperature(oof[a][0], y) for a in ARCHS}
    logps = [calibrated(oof[a][0], temperatures[a]) for a in ARCHS]
    feats = stack_features(logps)
    setups = setup_of_segment(seg)
    best_c, c_scores = select_c(feats, y, setups)
    meta = LogisticRegression(C=best_c, max_iter=5000).fit(feats, y)
    oof_metrics = {a: metrics(lp, y, seg) for a, lp in zip(ARCHS, logps)}
    oof_metrics["simple_average"] = metrics(average_logp(logps, [1, 1, 1]), y, seg)
    return {
        "temperatures": temperatures,
        "meta": meta,
        "c": best_c,
        "c_scores": c_scores,
        "oof_metrics": oof_metrics,
        "n_oof_windows": int(len(y)),
    }


def evaluate_seed(seed, fitted):
    out = {}
    val_acc = {}
    per_split = {}
    for split in ("val", "test"):
        loaded = {a: load_split(final_run(a, seed), split) for a in ARCHS}
        seg, y = loaded[ARCHS[0]][1], loaded[ARCHS[0]][2]
        for a in ARCHS[1:]:
            if not np.array_equal(loaded[a][1], seg):
                raise ValueError(f"{split} order mismatch for {a}")
        logps = {a: calibrated(loaded[a][0], fitted["temperatures"][a]) for a in ARCHS}
        per_split[split] = (logps, y, seg)
        if split == "val":
            val_acc = {a: float((logps[a].argmax(1) == y).mean()) for a in ARCHS}
    weights = [val_acc[a] for a in ARCHS]
    saved = {}
    for split, (logps, y, seg) in per_split.items():
        lps = [logps[a] for a in ARCHS]
        methods = {a: logps[a] for a in ARCHS}
        methods["simple_average"] = average_logp(lps, [1, 1, 1])
        methods["val_weighted_average"] = average_logp(lps, weights)
        methods["stacking"] = np.log(np.clip(fitted["meta"].predict_proba(stack_features(lps)), 1e-12, None))
        out[split] = {m: metrics(lp, y, seg) for m, lp in methods.items()}
        for m, lp in methods.items():
            saved[f"{split}_{m}"] = lp.astype(np.float32)
        saved[f"{split}_window_segment_id"] = seg
        saved[f"{split}_y"] = y
    np.savez_compressed(MODELS_DIR / "final" / f"ensemble_preds_s{seed}.npz", **saved)
    out["val_weights"] = dict(zip(ARCHS, (np.array(weights) / np.sum(weights)).tolist()))
    return out


def mean_std(per_seed, split, method, key):
    v = np.array([per_seed[s][split][method][key] for s in per_seed]) * 100
    return float(v.mean()), float(v.std())


def write_markdown(report):
    methods = ARCHS + ["simple_average", "val_weighted_average", "stacking"]
    label = {**NAMES, "simple_average": "Simple average", "val_weighted_average": "Validation-weighted average", "stacking": "Stacking (logistic regression)"}
    per_seed = report["per_seed"]
    lines = [
        "# Ensemble results",
        "",
        "Generated by `python -m src.ensemble.stack`. Mean +- std over seeds "
        + ", ".join(str(s) for s in per_seed)
        + ". The meta-learner and the temperatures were fitted on out-of-fold predictions for training setups 1-6 only. Setup 7 (validation) and setups 8-9 (test) were never used to fit them.",
        "",
        f"Meta-learner: multinomial logistic regression on {3 * N_CLASSES} calibrated log-probabilities, C = {report['meta_c']:g} (chosen by leave-one-setup-out log-loss over setups 1-6).",
        "",
        "| Method | Val window | Val recording | Test window | Test recording | Test recording macro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for m in methods:
        cells = []
        for split, key in (("val", "window_acc"), ("val", "segment_acc"), ("test", "window_acc"), ("test", "segment_acc"), ("test", "segment_macro_f1")):
            mu, sd = mean_std(per_seed, split, m, key)
            cells.append(f"{mu:.1f} +- {sd:.1f}")
        lines.append(f"| {label[m]} | " + " | ".join(cells) + " |")
    primary = report["primary_method"]
    lines += ["", f"Primary flat method (chosen on validation recording accuracy): **{label[primary]}**.", ""]
    lines += ["## Test recall per scenario, recording level, seed 42 (%)", "", "| Label | Scenario | " + " | ".join(label[m] for m in methods) + " |", "|---|---|" + "---|" * len(methods)]
    s42 = per_seed[str(SEEDS[0])]["test"]
    for k, name in enumerate(CLASS_NAMES):
        lines.append(f"| {k} | {name} | " + " | ".join(f"{100 * s42[m]['recall_segment'][k]:.0f}" for m in methods) + " |")
    lines += ["", f"## Test confusion matrix, {label[primary]}, recording level, seed 42", "", "Rows are true labels, columns predicted labels (0-16).", "", "| | " + " | ".join(str(k) for k in range(N_CLASSES)) + " |", "|---|" + "---|" * N_CLASSES]
    for k, row in enumerate(s42[primary]["confusion_segment"]):
        lines.append(f"| **{k}** | " + " | ".join(str(v) if v else "." for v in row) + " |")
    (RESULTS_DIR / "ensemble.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    fitted = fit_meta()
    per_seed = {str(s): evaluate_seed(s, fitted) for s in SEEDS}
    candidates = ["simple_average", "val_weighted_average", "stacking"]
    primary = max(candidates, key=lambda m: mean_std(per_seed, "val", m, "segment_acc")[0])
    report = {
        "protocol": {
            "meta_training_data": "out-of-fold predictions on training setups 1-6 (3 folds: setups {1,2}, {3,4}, {5,6} held out in turn)",
            "base_models": "final models trained on setups 1-6, checkpoint chosen on setup 7",
            "calibration": "per-model temperature scaling fitted on the same out-of-fold predictions",
            "val_weights": "validation-weighted average uses each model's setup-7 window accuracy as its weight",
            "primary_selection": "highest mean validation recording accuracy among the three ensembles",
            "recording_level": "sum of window log-probabilities over the 5 windows of a 60 s recording",
        },
        "temperatures": fitted["temperatures"],
        "meta_c": fitted["c"],
        "meta_c_loso_logloss": fitted["c_scores"],
        "oof_metrics": {k: {m: v[m] for m in ("window_acc", "segment_acc", "window_nll")} for k, v in fitted["oof_metrics"].items()},
        "n_oof_windows": fitted["n_oof_windows"],
        "primary_method": primary,
        "per_seed": per_seed,
        "summary": {
            m: {f"{split}_{key}": mean_std(per_seed, split, m, key) for split in ("val", "test") for key in ("window_acc", "segment_acc", "segment_macro_f1")}
            for m in ARCHS + candidates
        },
    }
    (RESULTS_DIR / "ensemble.json").write_text(json.dumps(report, indent=2))
    meta = fitted["meta"]
    np.savez(
        MODELS_DIR / "final" / "stacking_meta.npz",
        coef=meta.coef_,
        intercept=meta.intercept_,
        temperatures=np.array([fitted["temperatures"][a] for a in ARCHS]),
    )
    write_markdown(report)
    for m in ARCHS + candidates:
        print(m, {k: f"{v[0]:.1f}+-{v[1]:.1f}" for k, v in report["summary"][m].items()})
    print("primary", primary)


if __name__ == "__main__":
    main()
