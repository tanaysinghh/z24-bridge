import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocessing.pipeline import load_processed
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
    def __init__(self, segments, labels, window, crops_per_segment, device):
        self.segments = torch.from_numpy(segments).to(device)
        self.labels = torch.from_numpy(labels).to(device)
        self.window = window
        self.crops_per_segment = crops_per_segment
        self.device = device

    def epoch(self, batch_size, generator):
        n_seg, _, length = self.segments.shape
        seg_idx = torch.arange(n_seg, device=self.device).repeat(self.crops_per_segment)
        seg_idx = seg_idx[torch.randperm(len(seg_idx), generator=generator).to(self.device)]
        starts = torch.randint(0, length - self.window + 1, (len(seg_idx),), generator=generator).to(self.device)
        offsets = torch.arange(self.window, device=self.device)
        for i in range(0, len(seg_idx), batch_size):
            idx = seg_idx[i : i + batch_size]
            pos = starts[i : i + batch_size, None] + offsets[None, :]
            x = torch.gather(self.segments[idx, 0, :], 1, pos)[:, None, :]
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
            flip = torch.where(torch.rand(len(idx), 1, 1, generator=generator).to(self.device) < 0.5, -1.0, 1.0)
            yield x * flip, self.labels[idx]


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


def evaluate_split(model, data, split, device):
    scores = predict(model, data[f"{split}_X"], device)
    return evaluate_scores(scores, data[f"{split}_y"], data[f"{split}_window_segment_id"], segment_lookup(data, split))


def train(model_name, build_model, default_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="z24_ch22_w2000_s1000")
    parser.add_argument("--epochs", type=int, default=default_args.get("epochs", 80))
    parser.add_argument("--batch-size", type=int, default=default_args.get("batch_size", 64))
    parser.add_argument("--lr", type=float, default=default_args.get("lr", 1e-3))
    parser.add_argument("--weight-decay", type=float, default=default_args.get("weight_decay", 1e-2))
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--crops-per-segment", type=int, default=5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=model_name)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed(args.data)
    window = data["train_X"].shape[-1]
    model = build_model().to(device)
    n_params = count_parameters(model)
    print(f"{model_name}: {n_params:,} trainable parameters on {device}")

    sampler = RandomCropSampler(data["train_segments"], data["train_segment_y"], window, args.crops_per_segment, device)
    steps_per_epoch = int(np.ceil(len(data["train_segments"]) * args.crops_per_segment / args.batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    generator = torch.Generator().manual_seed(args.seed)

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
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(xb)
                loss = criterion(logits.float(), yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * len(yb)
                total_correct += (logits.argmax(1) == yb).sum().item()
                total_n += len(yb)
            val = evaluate_split(model, data, "val", device)
            row = {
                "epoch": epoch,
                "train_loss": total_loss / total_n,
                "train_crop_acc": total_correct / total_n,
                "val_window_acc": val["window_acc"],
                "val_segment_acc": val["segment_acc"],
            }
            history.append(row)
            print(json.dumps(row))
            if val["window_acc"] > best_val:
                best_val, best_epoch, stale = val["window_acc"], epoch, 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "args": vars(args)}, ckpt_path)
            else:
                stale += 1
                if stale >= args.patience:
                    break

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    results = {split: evaluate_split(model, data, split, device) for split in ("train", "val", "test")}
    payload = {
        "model": model_name,
        "parameters": n_params,
        "data": args.data,
        "args": vars(args),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_seconds": timer.elapsed,
        "device": str(device),
        "results": results,
        "history": history,
    }
    path = save_metrics(args.run_name, payload)
    for split in ("train", "val", "test"):
        r = results[split]
        print(f"{split}: window_acc={r['window_acc']:.4f} segment_acc={r['segment_acc']:.4f} macro_f1={r['window_macro_f1']:.4f}")
    print(f"checkpoint: {ckpt_path}\nmetrics: {path}")
    return payload
