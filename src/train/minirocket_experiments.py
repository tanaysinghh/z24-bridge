import argparse
import json
import warnings

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate

from src.preprocessing.pipeline import load_processed, zscore
from src.train.common import RESULTS_DIR, Timer, evaluate_scores, segment_lookup

ALPHAS = np.logspace(-1, 6, 15)


def features(transform, scaler, X, fit=False):
    raw = transform.fit_transform(X) if fit else transform.transform(X)
    raw = np.asarray(raw, dtype=np.float32)
    return scaler.fit_transform(raw) if fit else scaler.transform(raw)


def loso_alpha(feats, y, setups, alphas=ALPHAS):
    scores = {}
    for alpha in alphas:
        accs = []
        for held in np.unique(setups):
            tr, te = setups != held, setups == held
            ridge = RidgeClassifier(alpha=float(alpha)).fit(feats[tr], y[tr])
            accs.append(ridge.score(feats[te], y[te]))
        scores[float(alpha)] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "folds": [float(a) for a in accs]}
        print(f"alpha={alpha:.3g} loso={np.mean(accs):.4f}", flush=True)
    best = max(scores, key=lambda a: scores[a]["mean"])
    return best, scores


def augment(segments, labels, setups_seg, window, copies, jitter, scale, rng):
    n_seg, n_ch, length = segments.shape
    xs, ys, ss = [], [], []
    for _ in range(copies):
        starts = rng.integers(0, length - window + 1, n_seg)
        crops = np.stack([segments[i, :, s : s + window] for i, s in enumerate(starts)])
        crops = zscore(crops)
        crops = crops * (1 + rng.normal(0, scale, (n_seg, n_ch, 1))) + rng.normal(0, jitter, crops.shape)
        crops = crops * np.where(rng.random((n_seg, 1, 1)) < 0.5, -1.0, 1.0)
        xs.append(crops)
        ys.append(labels)
        ss.append(setups_seg)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)


def run(name, data_name, alpha_mode, seed, aug_copies=0, jitter=0.0, scale=0.0):
    data = load_processed(data_name)
    multi = data["train_X"].shape[1] > 1
    X_tr, y_tr, s_tr = data["train_X"].astype(np.float64), data["train_y"], data["train_setup"]
    if aug_copies:
        rng = np.random.default_rng(seed)
        seg_setups = (data["train_segment_id"] // 10) % 9 + 1
        Xa, ya, sa = augment(
            data["train_segments"].astype(np.float64), data["train_segment_y"], seg_setups,
            X_tr.shape[-1], aug_copies, jitter, scale, rng,
        )
        X_tr, y_tr, s_tr = np.concatenate([X_tr, Xa]), np.concatenate([y_tr, ya]), np.concatenate([s_tr, sa])
    transform = (MiniRocketMultivariate if multi else MiniRocket)(num_kernels=10000, random_state=seed, n_jobs=8)
    scaler = StandardScaler()
    with Timer() as timer:
        f_tr = features(transform, scaler, X_tr, fit=True)
        f_va = features(transform, scaler, data["val_X"].astype(np.float64))
        if alpha_mode == "loso":
            alpha, alpha_scores = loso_alpha(f_tr, y_tr, s_tr)
        else:
            alpha_scores = {}
            for a in ALPHAS:
                alpha_scores[float(a)] = float(RidgeClassifier(alpha=float(a)).fit(f_tr, y_tr).score(f_va, data["val_y"]))
            alpha = max(alpha_scores, key=alpha_scores.get)
        ridge = RidgeClassifier(alpha=float(alpha)).fit(f_tr, y_tr)
    results = {}
    for split, f in (("val", f_va),):
        results[split] = evaluate_scores(
            ridge.decision_function(f), data[f"{split}_y"], data[f"{split}_window_segment_id"], segment_lookup(data, split)
        )
    payload = {
        "model": "minirocket",
        "experiment": name,
        "data": data_name,
        "alpha_selection": alpha_mode,
        "ridge_alpha": float(alpha),
        "alpha_scores": alpha_scores,
        "aug_copies": aug_copies,
        "jitter": jitter,
        "scale": scale,
        "seed": seed,
        "train_seconds": timer.elapsed,
        "results": results,
    }
    out = RESULTS_DIR / "experiments" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"{name}: alpha={alpha:.3g} val win={results['val']['window_acc']:.4f} seg={results['val']['segment_acc']:.4f}", flush=True)
    return payload


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    plan = {
        "mr_val_alpha": dict(data_name="z24_ch22_w2000_s1000", alpha_mode="val"),
        "mr_loso_alpha": dict(data_name="z24_ch22_w2000_s1000", alpha_mode="loso"),
        "mr_aug": dict(data_name="z24_ch22_w2000_s1000", alpha_mode="loso", aug_copies=3, jitter=0.05, scale=0.1),
        "mr_mc": dict(data_name="z24_ch22-26_w2000_s1000", alpha_mode="loso"),
    }
    for name, kwargs in plan.items():
        if args.only and name not in args.only:
            continue
        run(name, seed=42, **kwargs)


if __name__ == "__main__":
    main()
