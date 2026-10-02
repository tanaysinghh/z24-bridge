import argparse
import json
import warnings

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocket

from src.preprocessing.pipeline import PROCESSED_DIR, PreprocessConfig, load_processed

ALPHAS = [1e2, 1e3, 1e4, 1e5]


def loso_minirocket(channel, setups, seed=42):
    data = load_processed(f"z24_ch{channel}_w2000_s1000")
    X = np.concatenate([data["train_X"], data["val_X"]]).astype(np.float64)
    y = np.concatenate([data["train_y"], data["val_y"]])
    s = np.concatenate([data["train_setup"], data["val_setup"]])
    win_seg = np.concatenate([data["train_window_segment_id"], data["val_window_segment_id"]])
    folds = {}
    for held in setups:
        tr, te = s != held, s == held
        mr = MiniRocket(num_kernels=10000, random_state=seed, n_jobs=-1)
        scaler = StandardScaler()
        f_tr = scaler.fit_transform(np.asarray(mr.fit_transform(X[tr]), dtype=np.float32))
        f_te = scaler.transform(np.asarray(mr.transform(X[te]), dtype=np.float32))
        folds[int(held)] = {}
        for alpha in ALPHAS:
            scores = RidgeClassifier(alpha=alpha).fit(f_tr, y[tr]).decision_function(f_te)
            seg, inv = np.unique(win_seg[te], return_inverse=True)
            seg_scores = np.zeros((len(seg), scores.shape[1]))
            np.add.at(seg_scores, inv, scores)
            seg_y = np.array([y[te][inv == k][0] for k in range(len(seg))])
            folds[int(held)][alpha] = {
                "window_acc": float((scores.argmax(1) == y[te]).mean()),
                "segment_acc": float((seg_scores.argmax(1) == seg_y).mean()),
            }
        print(channel, held, {a: round(v["window_acc"], 3) for a, v in folds[int(held)].items()}, flush=True)
    summary = {}
    for alpha in ALPHAS:
        w = [folds[h][alpha]["window_acc"] for h in setups]
        g = [folds[h][alpha]["segment_acc"] for h in setups]
        summary[alpha] = {"window_mean": float(np.mean(w)), "window_std": float(np.std(w)), "segment_mean": float(np.mean(g))}
    best = max(ALPHAS, key=lambda a: summary[a]["window_mean"])
    return {"channel": channel, "best_alpha": best, "best": summary[best], "by_alpha": summary, "folds": folds}


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, nargs="+", default=[1, 22, 24, 25, 26])
    args = parser.parse_args()
    cfg = PreprocessConfig()
    setups = cfg.train_setups + cfg.val_setups
    out_path = PROCESSED_DIR / "channel_selection_loso.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    for ch in args.channels:
        report[str(ch)] = loso_minirocket(ch, setups)
        out_path.write_text(json.dumps(report, indent=2, default=str))
    print("| channel | LOSO window acc (mean +- std) | LOSO segment acc | alpha |")
    print("|---|---|---|---|")
    for ch, r in sorted(report.items(), key=lambda kv: -kv[1]["best"]["window_mean"]):
        b = r["best"]
        print(f"| {ch} | {b['window_mean']:.3f} +- {b['window_std']:.3f} | {b['segment_mean']:.3f} | {r['best_alpha']:g} |")


if __name__ == "__main__":
    main()
