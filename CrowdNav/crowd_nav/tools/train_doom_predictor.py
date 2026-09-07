#!/usr/bin/env python3
"""Train the supervised doom predictor head from data/doom_dataset.pth."""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crowd_nav.policy.doom_predictor import DoomPredictor


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def roc_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model, features, labels, batch_size, device):
    model.eval()
    probs = []
    losses = []
    for start in range(0, len(labels), batch_size):
        end = min(len(labels), start + batch_size)
        x = features[start:end].to(device)
        y = labels[start:end].to(device)
        logits = model(x)
        losses.append(F.binary_cross_entropy_with_logits(logits, y).item() * len(y))
        probs.append(torch.sigmoid(logits).detach().cpu())
    probs = torch.cat(probs) if probs else torch.empty(0)
    loss = float(sum(losses) / max(1, len(labels)))
    auc = roc_auc(probs.numpy(), labels.cpu().numpy())
    pos_mean = float(probs[labels.cpu() >= 0.5].mean().item()) if (labels >= 0.5).any() else float("nan")
    neg_mean = float(probs[labels.cpu() < 0.5].mean().item()) if (labels < 0.5).any() else float("nan")
    return loss, auc, pos_mean, neg_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/doom_dataset.pth")
    parser.add_argument("--out", default="runs/mamba_vl/doom_predictor.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    data = safe_torch_load(args.data, map_location="cpu")
    train_features = data["train_features"].float()
    train_labels = data["train_labels"].float()
    val_features = data["val_features"].float()
    val_labels = data["val_labels"].float()
    if train_features.ndim != 2 or train_features.numel() == 0:
        raise RuntimeError("Empty train_features")
    if val_features.ndim != 2 or val_features.numel() == 0:
        raise RuntimeError("Empty val_features")

    input_dim = int(train_features.shape[1])
    hidden_dims = (128, 64)
    model = DoomPredictor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos = float(train_labels.sum().item())
    neg = float((train_labels < 0.5).sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    best = {"auc": -1.0, "state": None, "epoch": 0, "val_loss": None, "pos_mean": None, "neg_mean": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_n += len(y)

        val_loss, val_auc, pos_mean, neg_mean = evaluate(
            model, val_features, val_labels, args.batch_size, device
        )
        train_loss = total_loss / max(1, total_n)
        print(
            "[DOOM-TRAIN] epoch=%03d train_loss=%.4f val_loss=%.4f val_auc=%.4f pos_mean=%.4f neg_mean=%.4f"
            % (epoch, train_loss, val_loss, val_auc, pos_mean, neg_mean)
        )
        if np.isfinite(val_auc) and val_auc > best["auc"]:
            best.update({
                "auc": float(val_auc),
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": int(epoch),
                "val_loss": float(val_loss),
                "pos_mean": float(pos_mean),
                "neg_mean": float(neg_mean),
            })

    if best["state"] is None:
        best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best["auc"] = float("nan")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "model_state": best["state"],
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": float(args.dropout),
        "meta": {
            "best_epoch": best["epoch"],
            "val_auc": best["auc"],
            "val_loss": best["val_loss"],
            "val_pos_mean": best["pos_mean"],
            "val_neg_mean": best["neg_mean"],
            "data_meta": data.get("meta", {}),
        },
    }
    torch.save(payload, args.out)
    print(
        "[DOOM-TRAIN] saved %s best_epoch=%s val_auc=%.4f pos_mean=%.4f neg_mean=%.4f"
        % (args.out, best["epoch"], best["auc"], best["pos_mean"], best["neg_mean"])
    )


if __name__ == "__main__":
    main()
