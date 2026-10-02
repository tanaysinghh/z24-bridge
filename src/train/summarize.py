import json

import numpy as np

from src.train.common import RESULTS_DIR

MODELS = [("wavenet", "WaveNet"), ("minirocket", "MiniRocket"), ("inceptiontime", "InceptionTime")]
CLASS_NAMES = [
    "Reference 1",
    "Settlement system installed",
    "Pier -20 mm",
    "Pier -40 mm",
    "Pier -80 mm",
    "Pier -95 mm",
    "Foundation tilt",
    "Reference 2",
    "Spalling 12 m2",
    "Spalling 24 m2",
    "Landslide",
    "Concrete hinge failure",
    "2 anchor heads failed",
    "4 anchor heads failed",
    "2 tendons ruptured",
    "4 tendons ruptured",
    "6 tendons ruptured",
]


def pct(x):
    return f"{100 * x:.1f}"


def load():
    out = {}
    for key, name in MODELS:
        path = RESULTS_DIR / f"{key}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


def main_table(results):
    lines = [
        "| Model | Params | Train acc | Val acc | Test acc | Test segment acc | Test macro-F1 | Train time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        res = r["results"]
        params = f"{r['parameters']:,}" if name != "MiniRocket" else f"{r['parameters']:,} (ridge) + 10k fixed kernels"
        lines.append(
            f"| {name} | {params} | {pct(res['train']['window_acc'])} | {pct(res['val']['window_acc'])} | "
            f"{pct(res['test']['window_acc'])} | {pct(res['test']['segment_acc'])} | "
            f"{pct(res['test']['window_macro_f1'])} | {r['train_seconds'] / 60:.1f} min ({r['device']}) |"
        )
    return "\n".join(lines)


def segment_table(results):
    lines = ["| Model | Train segment acc | Val segment acc | Test segment acc | Test segment macro-F1 |", "|---|---|---|---|---|"]
    for name, r in results.items():
        res = r["results"]
        lines.append(
            f"| {name} | {pct(res['train']['segment_acc'])} | {pct(res['val']['segment_acc'])} | "
            f"{pct(res['test']['segment_acc'])} | {pct(res['test']['segment_macro_f1'])} |"
        )
    return "\n".join(lines)


def per_class_table(results):
    names = list(results)
    lines = ["| Label | Scenario | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    recalls = {}
    for name, r in results.items():
        cm = np.array(r["results"]["test"]["confusion_segment"])
        recalls[name] = cm.diagonal() / cm.sum(axis=1)
    for k, cname in enumerate(CLASS_NAMES):
        lines.append(f"| {k} | {cname} | " + " | ".join(pct(recalls[n][k]) for n in names) + " |")
    return "\n".join(lines)


def seed_table(seeds=(42, 1, 2)):
    lines = [
        "| Model | Seeds | Val acc (mean +- std) | Test acc (mean +- std) | Test segment acc (mean +- std) |",
        "|---|---|---|---|---|",
    ]
    for key, name in MODELS:
        runs = []
        for seed in seeds:
            path = RESULTS_DIR / (f"{key}.json" if seed == 42 else f"{key}_seed{seed}.json")
            if path.exists():
                runs.append(json.loads(path.read_text())["results"])
        if not runs:
            continue
        cells = []
        for split, metric in (("val", "window_acc"), ("test", "window_acc"), ("test", "segment_acc")):
            v = np.array([r[split][metric] for r in runs]) * 100
            cells.append(f"{v.mean():.1f} +- {v.std():.1f}")
        lines.append(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    results = load()
    text = "\n\n".join(
        [
            "## Window-level accuracy (%)",
            main_table(results),
            "## Segment-level accuracy (%)",
            segment_table(results),
            "## Seed variability (%)",
            seed_table(),
            "## Test recall per scenario, segment level (%)",
            per_class_table(results),
        ]
    )
    path = RESULTS_DIR / "summary_tables.md"
    path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
