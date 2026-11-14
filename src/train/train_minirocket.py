import argparse
import pickle
import warnings

import numpy as np
from sklearn.linear_model import RidgeClassifier

from src.models.minirocket import MiniRocketClassifier
from src.preprocessing.pipeline import load_processed, zscore
from src.taxonomy import setup_of_segment
from src.train.common import MODELS_DIR, Timer, evaluate_scores, save_metrics, segment_lookup, set_seed


def loso_alpha(clf, feats, y, setups):
    scores = {}
    for alpha in clf.alphas:
        accs = []
        for held in np.unique(setups):
            tr, te = setups != held, setups == held
            accs.append(RidgeClassifier(alpha=float(alpha)).fit(feats[tr], y[tr]).score(feats[te], y[te]))
        scores[float(alpha)] = float(np.mean(accs))
    return max(scores, key=scores.get), scores


def split_arrays(data, split, setups=None):
    keep = np.ones(len(data[f"{split}_y"]), dtype=bool)
    if setups is not None:
        keep = np.isin(data[f"{split}_setup"], setups)
    seg_keep = np.isin(data[f"{split}_segment_id"], np.unique(data[f"{split}_window_segment_id"][keep]))
    return {
        "X": data[f"{split}_X"][keep],
        "y": data[f"{split}_y"][keep],
        "setup": data[f"{split}_setup"][keep],
        "window_segment_id": data[f"{split}_window_segment_id"][keep],
        "lookup": dict(zip(data[f"{split}_segment_id"][seg_keep].tolist(), data[f"{split}_segment_y"][seg_keep].tolist())),
    }


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="z24_ch22_w2000_s1000")
    parser.add_argument("--num-kernels", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="minirocket")
    parser.add_argument("--train-stride", type=int, default=None)
    parser.add_argument("--alpha-mode", choices=["val", "loso"], default="val")
    parser.add_argument("--train-setups", type=int, nargs="+", default=None)
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data = load_processed(args.data)
    train_setups = args.train_setups or [1, 2, 3, 4, 5, 6]
    held_setups = [s for s in [1, 2, 3, 4, 5, 6] if s not in train_setups]
    tr = split_arrays(data, "train", train_setups)
    train_X, train_y = tr["X"], tr["y"]
    if args.train_stride is not None:
        segs, window = data["train_segments"][:, 0, :], data["train_X"].shape[-1]
        starts = np.arange(0, segs.shape[1] - window + 1, args.train_stride)
        train_X = zscore(np.stack([segs[:, s : s + window] for s in starts], axis=1)).reshape(-1, 1, window)
        train_y = np.repeat(data["train_segment_y"], len(starts))
    splits = {"train": tr, "val": split_arrays(data, "val")}
    if args.eval_test:
        splits["test"] = split_arrays(data, "test")
    if held_setups:
        splits["oof"] = split_arrays(data, "train", held_setups)

    clf = MiniRocketClassifier(num_kernels=args.num_kernels, random_state=args.seed)
    with Timer() as timer:
        if args.alpha_mode == "val":
            clf.fit(train_X.astype(np.float64), train_y, splits["val"]["X"].astype(np.float64), splits["val"]["y"])
        else:
            feats = clf.fit_features(train_X.astype(np.float64))
            alpha, clf.alpha_scores = loso_alpha(clf, feats, train_y, tr["setup"])
            clf.ridge = RidgeClassifier(alpha=float(alpha)).fit(feats, train_y)
    print(f"minirocket: {clf.n_features} features, ridge alpha={clf.alpha:.4g} ({args.alpha_mode}), fit {timer.elapsed:.1f}s")

    results, preds = {}, {}
    for split, sub in splits.items():
        scores = clf.decision_function(sub["X"].astype(np.float64))
        results[split] = evaluate_scores(scores, sub["y"], sub["window_segment_id"], sub["lookup"])
        preds[f"{split}_scores"] = scores.astype(np.float32)
        preds[f"{split}_window_segment_id"] = sub["window_segment_id"]
        preds[f"{split}_y"] = sub["y"]
        r = results[split]
        print(f"{split}: window_acc={r['window_acc']:.4f} segment_acc={r['segment_acc']:.4f} macro_f1={r['window_macro_f1']:.4f}")

    out_dir = MODELS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(clf, f)
    np.savez_compressed(out_dir / "preds.npz", **preds)
    payload = {
        "model": "minirocket",
        "parameters": int(clf.ridge.coef_.size + clf.ridge.intercept_.size),
        "n_features": int(clf.n_features),
        "ridge_alpha": float(clf.alpha),
        "alpha_selection": args.alpha_mode,
        "alpha_scores": clf.alpha_scores,
        "data": args.data,
        "train_setups": train_setups,
        "args": vars(args),
        "train_seconds": timer.elapsed,
        "device": "cpu",
        "results": results,
    }
    path = save_metrics(args.run_name, payload)
    print(f"checkpoint: {out_dir / 'model.pkl'}\nmetrics: {path}")


if __name__ == "__main__":
    main()
