"""Evaluate a trained HiCon checkpoint on a chronological CSV test split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .model import TemporalSpectralForecaster, InstanceRefinement
from .data import load_csv, make_datasets
from .engine import evaluate_model, loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["cfg"]
    values, columns = load_csv(args.data, ",".join(ckpt["columns"]))
    if columns != ckpt["columns"]:
        raise ValueError("CSV columns differ from the trained model")
    if len(values) != ckpt["n_rows"]:
        raise ValueError("Number of data rows differs from original training CSV")
    scaler = (np.asarray(ckpt["scaler_mean"], np.float32), np.asarray(ckpt["scaler_std"], np.float32))
    datasets, _, _ = make_datasets(values, cfg["lookback"], cfg["horizon"],
        cfg["train_ratio"], cfg["val_ratio"], scaler=scaler,
        split_bounds=ckpt["split_bounds"])
    network_fields = ("lookback", "horizon", "d_model", "d_ff", "n_heads", "n_experts", "patch_len", "dropout")
    backbone = TemporalSpectralForecaster(**{k: cfg[k] for k in network_fields}).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    gate = InstanceRefinement(cfg["d_model"], cfg["gate_hidden"], cfg["epsilon"]).to(device)
    gate.load_state_dict(ckpt["gate"])
    test_loader = loader(datasets[2], cfg["batch_size"])
    score = evaluate_model(backbone, test_loader, device,
        profile=ckpt["profile"], alpha=ckpt["alpha"],
        gate=gate if ckpt["gate_enabled"] else None, scaler=scaler)
    print(json.dumps({"test": score, "alpha": ckpt["alpha"],
                      "gate_enabled": ckpt["gate_enabled"]}, indent=2))


if __name__ == "__main__":
    main()
