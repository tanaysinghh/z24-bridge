import json

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.models.minirocket import MiniRocketClassifier
from src.preprocessing.pipeline import load_processed
from src.train.common import RESULTS_DIR, evaluate_scores


def main():
    data = load_processed()
    X = np.concatenate([data["train_X"], data["val_X"]]).astype(np.float64)
    y = np.concatenate([data["train_y"], data["val_y"]])
    win_seg = np.concatenate([data["train_window_segment_id"], data["val_window_segment_id"]])
    seg_ids = np.concatenate([data["train_segment_id"], data["val_segment_id"]])
    seg_y = np.concatenate([data["train_segment_y"], data["val_segment_y"]])
    lookup = dict(zip(seg_ids.tolist(), seg_y.tolist()))

    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    fit_seg, hold_seg = next(outer.split(seg_ids, seg_y))
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr_seg, va_seg = next(inner.split(fit_seg, seg_y[fit_seg]))
    tr = np.isin(win_seg, seg_ids[fit_seg][tr_seg])
    va = np.isin(win_seg, seg_ids[fit_seg][va_seg])
    ho = np.isin(win_seg, seg_ids[hold_seg])

    clf = MiniRocketClassifier(random_state=42).fit(X[tr], y[tr], X[va], y[va])
    res = evaluate_scores(clf.decision_function(X[ho]), y[ho], win_seg[ho], lookup)
    res.pop("confusion_segment")
    payload = {
        "description": "MiniRocket, segment-grouped stratified random split over setups 1-7 (test setups 8-9 excluded); diagnostic only",
        "ridge_alpha": float(clf.alpha),
        "holdout": res,
    }
    (RESULTS_DIR / "diagnostic_random_split.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
