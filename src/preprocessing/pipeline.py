import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "z24"
PROCESSED_DIR = ROOT / "data" / "processed"

N_SCENARIOS = 17
N_SETUPS = 9
N_SEGMENTS = 10
N_CHANNELS = 27
SEGMENT_LEN = 6000


@dataclass
class PreprocessConfig:
    channel: int = 22
    fs: float = 100.0
    band_low: float = 0.5
    band_high: float = 30.0
    filter_order: int = 4
    window: int = 2000
    stride: int = 1000
    train_setups: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    val_setups: list = field(default_factory=lambda: [7])
    test_setups: list = field(default_factory=lambda: [8, 9])

    @property
    def name(self):
        return f"z24_ch{self.channel}_w{self.window}_s{self.stride}"


def load_raw(raw_dir=RAW_DIR):
    inputs = np.load(raw_dir / "inputs.npy", mmap_mode="r")
    labels = np.load(raw_dir / "labels.npy")
    if inputs.shape != (N_SCENARIOS * N_SETUPS * N_SEGMENTS, N_CHANNELS, SEGMENT_LEN):
        raise ValueError(f"unexpected inputs shape {inputs.shape}")
    if labels.shape != (inputs.shape[0],) or len(np.unique(labels)) != N_SCENARIOS:
        raise ValueError(f"unexpected labels shape {labels.shape}")
    return inputs, labels


def sample_index():
    idx = np.arange(N_SCENARIOS * N_SETUPS * N_SEGMENTS)
    return {
        "scenario": idx // (N_SETUPS * N_SEGMENTS),
        "setup": (idx // N_SEGMENTS) % N_SETUPS + 1,
        "segment": idx % N_SEGMENTS,
    }


def select_channel(inputs, channel):
    return np.asarray(inputs[:, channel, :], dtype=np.float64)


def bandpass_records(segments, cfg):
    sos = butter(cfg.filter_order, [cfg.band_low, cfg.band_high], btype="bandpass", fs=cfg.fs, output="sos")
    records = segments.reshape(N_SCENARIOS * N_SETUPS, N_SEGMENTS * SEGMENT_LEN)
    records = records - records.mean(axis=1, keepdims=True)
    filtered = sosfiltfilt(sos, records, axis=1)
    return filtered.reshape(-1, SEGMENT_LEN)


def zscore(x, axis=-1, eps=1e-8):
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)


def window_starts(cfg):
    return np.arange(0, SEGMENT_LEN - cfg.window + 1, cfg.stride)


def make_windows(segments, cfg):
    starts = window_starts(cfg)
    windows = np.stack([segments[:, s : s + cfg.window] for s in starts], axis=1)
    return zscore(windows), starts


def split_masks(setups, cfg):
    return {
        "train": np.isin(setups, cfg.train_setups),
        "val": np.isin(setups, cfg.val_setups),
        "test": np.isin(setups, cfg.test_setups),
    }


def build(cfg, raw_dir=RAW_DIR, out_dir=PROCESSED_DIR):
    inputs, labels = load_raw(raw_dir)
    meta = sample_index()
    if not np.array_equal(labels, meta["scenario"]):
        raise ValueError("labels do not follow scenario-major ordering")
    segments = bandpass_records(select_channel(inputs, cfg.channel), cfg)
    windows, starts = make_windows(segments, cfg)
    n_win = windows.shape[1]
    masks = split_masks(meta["setup"], cfg)

    arrays = {}
    for split, mask in masks.items():
        seg_ids = np.flatnonzero(mask)
        arrays[f"{split}_segments"] = segments[mask][:, None, :].astype(np.float32)
        arrays[f"{split}_segment_y"] = labels[mask].astype(np.int64)
        arrays[f"{split}_segment_id"] = seg_ids
        arrays[f"{split}_X"] = windows[mask].reshape(-1, 1, cfg.window).astype(np.float32)
        arrays[f"{split}_y"] = np.repeat(labels[mask], n_win).astype(np.int64)
        arrays[f"{split}_window_segment_id"] = np.repeat(seg_ids, n_win)
        arrays[f"{split}_setup"] = np.repeat(meta["setup"][mask], n_win)
    arrays["window_starts"] = starts

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.name}.npz"
    np.savez_compressed(out_path, **arrays)
    summary = {
        "config": asdict(cfg),
        "windows_per_segment": int(n_win),
        "counts": {s: {"segments": int(m.sum()), "windows": int(m.sum() * n_win)} for s, m in masks.items()},
    }
    (out_dir / f"{cfg.name}.json").write_text(json.dumps(summary, indent=2))
    return out_path, summary


def load_processed(name="z24_ch22_w2000_s1000", out_dir=PROCESSED_DIR):
    path = out_dir / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run python -m src.preprocessing.pipeline first")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=22)
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=1000)
    args = parser.parse_args()
    cfg = PreprocessConfig(channel=args.channel, window=args.window, stride=args.stride)
    out_path, summary = build(cfg)
    print(out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
