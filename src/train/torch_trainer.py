import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocessing.pipeline import load_processed
from src.taxonomy import CLASS_TO_GROUP_INDEX, GROUP_MEMBERS, GROUP_NAMES, setup_of_segment
from src.train.common import (
    MODELS_DIR,
    Timer,
    count_parameters,
    evaluate_scores,
    save_metrics,
    segment_lookup,
    set_seed,
)


class RandomCropSampler:
    def __init__(self, segments, labels, window, crops_per_segment, device, jitter=0.0, scale=0.0, time_warp=0.0):
        self.segments = torch.from_numpy(segments).to(device)
        self.labels = torch.from_numpy(labels).to(device)
        self.window = window
        self.crops_per_segment = crops_per_segment
        self.device = device
        self.jitter = jitter
        self.scale = scale
        self.time_warp = time_warp

    def crop(self, idx, generator):
        n = len(idx)
        length = self.segments.shape[-1]
        offsets = torch.arange(self.window, device=self.device, dtype=torch.float32)
        if self.time_warp > 0:
            factor = 1.0 + (torch.rand(n, 1, generator=generator) * 2 - 1).to(self.device) * self.time_warp
        else:
            factor = torch.ones(n, 1, device=self.device)
        span = (self.window - 1) * factor
        start = torch.rand(n, 1, generator=generator).to(self.device) * (length - 1 - span)
        if self.time_warp <= 0:
            start = start.floor()
        pos = start + offsets[None, :] * factor
        lo = pos.floor().long().clamp(0, length - 1)
        hi = (lo + 1).clamp(max=length - 1)
        frac = (pos - lo.float())[:, None, :]
        seg = self.segments[idx]
        n_ch = seg.shape[1]
        lo_v = torch.gather(seg, 2, lo[:, None, :].expand(-1, n_ch, -1))
        hi_v = torch.gather(seg, 2, hi[:, None, :].expand(-1, n_ch, -1))
        x = lo_v * (1 - frac) + hi_v * frac
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        flip = torch.where(torch.rand(n, 1, 1, generator=generator).to(self.device) < 0.5, -1.0, 1.0)
        x = x * flip
        if self.scale > 0:
            x = x * (1.0 + torch.randn(n, n_ch, 1, generator=generator).to(self.device) * self.scale)
        if self.jitter > 0:
            x = x + torch.randn(x.shape, generator=generator).to(self.device) * self.jitter
        return x

    def epoch(self, batch_size, generator):
        n_seg = self.segments.shape[0]
        seg_idx = torch.arange(n_seg, device=self.device).repeat(self.crops_per_segment)
        seg_idx = seg_idx[torch.randperm(len(seg_idx), generator=generator).to(self.device)]
        for i in range(0, len(seg_idx), batch_size):
            idx = seg_idx[i : i + batch_size]
            yield self.crop(idx, generator), self.labels[idx]


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(xb)
        out.append(F.log_softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def task_spec(task):
    if task == "flat":
        return np.arange(17), np.arange(17), 17
    if task == "group":
        return np.arange(17), np.array(CLASS_TO_GROUP_INDEX), len(GROUP_NAMES)
    if task.startswith("specialist:"):
        members = np.array(GROUP_MEMBERS[task.split(":", 1)[1]])
        mapping = np.full(17, -1)
        mapping[members] = np.arange(len(members))
        return members, mapping, len(members)
    raise ValueError(task)


def subset_split(data, split, classes, mapping, setups=None):
    seg_y = data[f"{split}_segment_y"]
    seg_id = data[f"{split}_segment_id"]
    seg_keep = np.isin(seg_y, classes)
    if setups is not None:
        seg_keep &= np.isin(setup_of_segment(seg_id), setups)
    win_keep = np.isin(data[f"{split}_window_segment_id"], seg_id[seg_keep])
    return {
        "X": data[f"{split}_X"][win_keep],
        "y": mapping[data[f"{split}_y"][win_keep]],
        "window_segment_id": data[f"{split}_window_segment_id"][win_keep],
        "segments": data[f"{split}_segments"][seg_keep],
        "segment_y": mapping[seg_y[seg_keep]],
        "segment_id": seg_id[seg_keep],
    }


def evaluate_subset(model, sub, device, n_classes):
    scores = predict(model, sub["X"], device)
    lookup = dict(zip(sub["segment_id"].tolist(), sub["segment_y"].tolist()))
    return scores, evaluate_scores(scores, sub["y"], sub["window_segment_id"], lookup, n_classes)


def parse_args(model_name, default_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="z24_ch22_w2000_s1000")
    parser.add_argument("--epochs", type=int, default=default_args.get("epochs", 80))
    parser.add_argument("--batch-size", type=int, default=default_args.get("batch_size", 64))
    parser.add_argument("--lr", type=float, default=default_args.get("lr", 1e-3))
    parser.add_argument("--weight-decay", type=float, default=default_args.get("weight_decay", 1e-2))
    parser.add_argument("--dropout", type=float, default=default_args.get("dropout", 0.1))
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--crops-per-segment", type=int, default=5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--time-warp", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--task", default="flat")
    parser.add_argument("--train-setups", type=int, nargs="+", default=None)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=model_name)
    return parser.parse_args()


def train(model_name, build_model, default_args):
    args = parse_args(model_name, default_args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed(args.data)
    classes, mapping, n_classes = task_spec(args.task)
    window = data["train_X"].shape[-1]
    in_channels = data["train_X"].shape[1]

    train_setups = args.train_setups or [1, 2, 3, 4, 5, 6]
    held_setups = [s for s in [1, 2, 3, 4, 5, 6] if s not in train_setups]
    splits = {
        "train": subset_split(data, "train", classes, mapping, train_setups),
        "val": subset_split(data, "val", classes, mapping),
    }
    if args.eval_test:
        splits["test"] = subset_split(data, "test", classes, mapping)
    if held_setups:
        splits["oof"] = subset_split(data, "train", classes, mapping, held_setups)

    model = build_model(in_channels, n_classes, args).to(device)
    n_params = count_parameters(model)
    print(f"{model_name}: {n_params:,} trainable parameters on {device}, {in_channels} channels, {n_classes} classes, task {args.task}")

    tr = splits["train"]
    sampler = RandomCropSampler(
        tr["segments"], tr["segment_y"], window, args.crops_per_segment, device, args.jitter, args.scale, args.time_warp
    )
    steps_per_epoch = int(np.ceil(len(tr["segments"]) * args.crops_per_segment / args.batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    generator = torch.Generator().manual_seed(args.seed)
    beta = torch.distributions.Beta(args.mixup, args.mixup) if args.mixup > 0 else None

    out_dir = MODELS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    history = []
    best_val, best_epoch, stale = -1.0, -1, 0

    with Timer() as timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, total_correct, total_n = 0.0, 0, 0
            for xb, yb in sampler.epoch(args.batch_size, generator):
                if beta is not None:
                    lam = float(beta.sample())
                    perm = torch.randperm(len(yb), generator=generator).to(device)
                    xb = lam * xb + (1 - lam) * xb[perm]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(xb)
                logits = logits.float()
                if beta is not None:
                    loss = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb[perm])
                else:
                    loss = criterion(logits, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * len(yb)
                total_correct += (logits.argmax(1) == yb).sum().item()
                total_n += len(yb)
            _, val = evaluate_subset(model, splits["val"], device, n_classes)
            row = {
                "epoch": epoch,
                "train_loss": total_loss / total_n,
                "train_crop_acc": total_correct / total_n,
                "val_window_acc": val["window_acc"],
                "val_segment_acc": val["segment_acc"],
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if val["window_acc"] > best_val:
                best_val, best_epoch, stale = val["window_acc"], epoch, 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "args": vars(args),
                        "in_channels": in_channels,
                        "n_classes": n_classes,
                    },
                    ckpt_path,
                )
            else:
                stale += 1
                if stale >= args.patience:
                    break

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    results, preds = {}, {}
    for split, sub in splits.items():
        scores, results[split] = evaluate_subset(model, sub, device, n_classes)
        preds[f"{split}_logp"] = scores.astype(np.float32)
        preds[f"{split}_window_segment_id"] = sub["window_segment_id"]
        preds[f"{split}_y"] = sub["y"]
    if args.task != "flat":
        for split in ("val", "test") if args.eval_test else ("val",):
            preds[f"all_{split}_logp"] = predict(model, data[f"{split}_X"], device).astype(np.float32)
            preds[f"all_{split}_window_segment_id"] = data[f"{split}_window_segment_id"]
            preds[f"all_{split}_y"] = data[f"{split}_y"]
    np.savez_compressed(out_dir / "preds.npz", **preds)
    payload = {
        "model": model_name,
        "parameters": n_params,
        "data": args.data,
        "in_channels": in_channels,
        "n_classes": n_classes,
        "task": args.task,
        "train_setups": train_setups,
        "args": vars(args),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_seconds": timer.elapsed,
        "device": str(device),
        "results": results,
        "history": history,
    }
    path = save_metrics(args.run_name, payload)
    for split, r in results.items():
        print(f"{split}: window_acc={r['window_acc']:.4f} segment_acc={r['segment_acc']:.4f} macro_f1={r['window_macro_f1']:.4f}")
    print(f"checkpoint: {ckpt_path}\nmetrics: {path}")
    return payload
