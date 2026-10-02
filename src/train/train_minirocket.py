import argparse
import pickle

import numpy as np

from src.models.minirocket import MiniRocketClassifier
from src.preprocessing.pipeline import load_processed, zscore
from src.train.common import MODELS_DIR, Timer, evaluate_scores, save_metrics, segment_lookup, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="z24_ch22_w2000_s1000")
    parser.add_argument("--num-kernels", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="minirocket")
    parser.add_argument("--train-stride", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    data = load_processed(args.data)
    train_X, train_y = data["train_X"], data["train_y"]
    if args.train_stride is not None:
        segs, window = data["train_segments"][:, 0, :], data["train_X"].shape[-1]
        starts = np.arange(0, segs.shape[1] - window + 1, args.train_stride)
        train_X = zscore(np.stack([segs[:, s : s + window] for s in starts], axis=1)).reshape(-1, 1, window)
        train_y = np.repeat(data["train_segment_y"], len(starts))
    clf = MiniRocketClassifier(num_kernels=args.num_kernels, random_state=args.seed)
    with Timer() as timer:
        clf.fit(
            train_X.astype(np.float64),
            train_y,
            data["val_X"].astype(np.float64),
            data["val_y"],
        )
    print(f"minirocket: {clf.n_features} features, ridge alpha={clf.alpha:.4g}, fit {timer.elapsed:.1f}s")

    results = {}
    for split in ("train", "val", "test"):
        scores = clf.decision_function(data[f"{split}_X"].astype(np.float64))
        results[split] = evaluate_scores(
            scores, data[f"{split}_y"], data[f"{split}_window_segment_id"], segment_lookup(data, split)
        )
        r = results[split]
        print(f"{split}: window_acc={r['window_acc']:.4f} segment_acc={r['segment_acc']:.4f} macro_f1={r['window_macro_f1']:.4f}")

    out_dir = MODELS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(clf, f)
    payload = {
        "model": "minirocket",
        "parameters": int(clf.ridge.coef_.size + clf.ridge.intercept_.size),
        "n_features": int(clf.n_features),
        "ridge_alpha": float(clf.alpha),
        "val_window_acc_by_alpha": clf.alpha_scores,
        "data": args.data,
        "args": vars(args),
        "train_seconds": timer.elapsed,
        "device": "cpu",
        "results": results,
    }
    path = save_metrics(args.run_name, payload)
    print(f"checkpoint: {out_dir / 'model.pkl'}\nmetrics: {path}")


if __name__ == "__main__":
    main()
