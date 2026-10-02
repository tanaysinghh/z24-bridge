import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "docs" / "results"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def segment_scores(window_scores, window_segment_id):
    seg_ids, inverse = np.unique(window_segment_id, return_inverse=True)
    summed = np.zeros((len(seg_ids), window_scores.shape[1]))
    np.add.at(summed, inverse, window_scores)
    return seg_ids, summed


def evaluate_scores(scores, y, window_segment_id, segment_y_lookup):
    pred = scores.argmax(axis=1)
    seg_ids, seg_scores = segment_scores(scores, window_segment_id)
    seg_pred = seg_scores.argmax(axis=1)
    seg_true = np.array([segment_y_lookup[s] for s in seg_ids])
    return {
        "window_acc": float((pred == y).mean()),
        "window_macro_f1": float(f1_score(y, pred, average="macro")),
        "segment_acc": float((seg_pred == seg_true).mean()),
        "segment_macro_f1": float(f1_score(seg_true, seg_pred, average="macro")),
        "n_windows": int(len(y)),
        "n_segments": int(len(seg_ids)),
        "confusion_segment": confusion_matrix(seg_true, seg_pred, labels=np.arange(17)).tolist(),
    }


def segment_lookup(data, split):
    return dict(zip(data[f"{split}_segment_id"].tolist(), data[f"{split}_segment_y"].tolist()))


def save_metrics(model_name, payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{model_name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
