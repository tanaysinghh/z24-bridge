import json

from src.train.common import RESULTS_DIR

DESCRIPTIONS = {
    "base": "Baseline protocol (crop + sign flip only), single channel 22",
    "mc": "Multi-channel input, reference channels 22-26",
    "jitscale": "+ Gaussian jitter (sd 0.05) and amplitude scaling (sd 0.1)",
    "mixup": "+ mixup (alpha 0.2)",
    "twarp": "+ time warp (+-2 %)",
    "lr3e-3": "Learning rate 3e-3 (baseline 1e-3)",
    "lr3e-4": "Learning rate 3e-4 (baseline 1e-3)",
    "reg": "Stronger regularisation: weight decay 0.05, dropout 0.3",
    "wd5e-2": "Weight decay 0.05",
}
MR_DESCRIPTIONS = {
    "mr_val_alpha": "Ridge alpha chosen on setup 7 (Review 2 protocol)",
    "mr_loso_alpha": "Ridge alpha by leave-one-setup-out CV over setups 1-6",
    "mr_aug": "LOSO alpha + 3 augmented crops per segment (jitter, scaling, sign flip)",
    "mr_mc": "LOSO alpha, multivariate MiniRocket on channels 22-26",
}


def pct(x):
    return f"{100 * x:.1f}"


def torch_rows(prefix):
    rows = []
    for path in sorted((RESULTS_DIR / "exp").glob(f"{prefix}_*.json")):
        r = json.loads(path.read_text())
        key = path.stem[len(prefix) + 1 :]
        res = r["results"]
        rows.append(
            {
                "key": key,
                "desc": DESCRIPTIONS.get(key, key),
                "train": res["train"]["window_acc"],
                "val_w": res["val"]["window_acc"],
                "val_s": res["val"]["segment_acc"],
                "best_epoch": r["best_epoch"],
                "minutes": r["train_seconds"] / 60,
            }
        )
    return rows


def table(rows, base_key="base"):
    base = next((r for r in rows if r["key"] == base_key), None)
    lines = ["| Run | Change | Train window | Val window | Val recording | Delta val window vs base | Best epoch |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        delta = f"{100 * (r['val_w'] - base['val_w']):+.1f}" if base else ""
        lines.append(f"| {r['key']} | {r['desc']} | {pct(r['train'])} | {pct(r['val_w'])} | {pct(r['val_s'])} | {delta} | {r['best_epoch']} |")
    return "\n".join(lines)


def minirocket_table():
    lines = ["| Run | Change | Ridge alpha | Val window | Val recording |", "|---|---|---|---|---|"]
    for key, desc in MR_DESCRIPTIONS.items():
        path = RESULTS_DIR / "experiments" / f"{key}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        v = r["results"]["val"]
        lines.append(f"| {key} | {desc} | {r['ridge_alpha']:.3g} | {pct(v['window_acc'])} | {pct(v['segment_acc'])} |")
    return "\n".join(lines)


def main():
    text = "\n\n".join(
        [
            "## WaveNet screening (seed 42, 50 epochs)",
            table(torch_rows("wn")),
            "## InceptionTime screening (seed 42, 50 epochs)",
            table(torch_rows("it")),
            "## MiniRocket",
            minirocket_table(),
        ]
    )
    (RESULTS_DIR / "experiments_tables.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
