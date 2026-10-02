import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESULTS = DOCS / "results"

MODELS = [
    ("wavenet", "WaveNet"),
    ("minirocket", "MiniRocket"),
    ("inceptiontime", "InceptionTime"),
]
SEEDS = {"": 42, "_seed1": 1, "_seed2": 2}
SPLITS = [("train", "Train", "setups 1–6"), ("val", "Validation", "setup 7"), ("test", "Test", "setups 8–9")]

DATASET_FACTS = [
    ("1963", "Built", "Koppigen–Utzenstorf, Switzerland"),
    ("58 m", "Length", "Spans 14 + 30 + 14 m"),
    ("17", "Damage scenarios", "Progressive damage tests, 1998"),
    ("27", "Accelerometers", "Per setup"),
    ("9", "Sensor setups", "Per scenario"),
    ("1,530", "Recordings", "17 × 9 × 10"),
    ("100 Hz", "Sampling rate", "Verified from modal peaks"),
    ("60 s", "Per recording", "6,000 samples"),
]

SHORT_NAMES = [
    "Reference 1",
    "Settlement system",
    "Pier −20 mm",
    "Pier −40 mm",
    "Pier −80 mm",
    "Pier −95 mm",
    "Foundation tilt",
    "Reference 2",
    "Spalling 12 m²",
    "Spalling 24 m²",
    "Landslide",
    "Concrete hinge failure",
    "2 anchor heads failed",
    "4 anchor heads failed",
    "2 tendons ruptured",
    "4 tendons ruptured",
    "6 tendons ruptured",
]


def load_json(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def scenario_names():
    text = (DOCS / "dataset.md").read_text(encoding="utf-8")
    section = text.split("## 3. Classes", 1)[1].split("##", 1)[0]
    rows = re.findall(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*$", section, flags=re.M)
    return [name for _, name in sorted(rows, key=lambda r: int(r[0]))]


def model_summary():
    out = []
    for key, name in MODELS:
        runs = [load_json(f"{key}{suffix}") for suffix in SEEDS]
        primary = runs[0]

        def stat(split, metric):
            v = np.array([r["results"][split][metric] for r in runs]) * 100
            return float(v.mean()), float(v.std())

        out.append(
            {
                "key": key,
                "name": name,
                "params": primary["parameters"],
                "n_features": primary.get("n_features"),
                "seeds": len(runs),
                "test_window": stat("test", "window_acc"),
                "test_segment": stat("test", "segment_acc"),
                "val_window": stat("val", "window_acc"),
                "val_segment": stat("val", "segment_acc"),
                "train_minutes": primary["train_seconds"] / 60,
                "device": primary["device"],
                "recall": np.array(primary["results"]["test"]["confusion_segment"], dtype=float),
            }
        )
    for m in out:
        cm = m["recall"]
        m["recall"] = cm.diagonal() / cm.sum(axis=1) * 100
    return out


def split_counts():
    primary = load_json("wavenet")["results"]
    return {
        split: np.array(primary[split]["confusion_segment"]).sum(axis=1).astype(int) for split, _, _ in SPLITS
    }


def chance_level(n_classes=17):
    return 100 / n_classes
