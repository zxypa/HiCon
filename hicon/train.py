"""Command-line training and test evaluation: python -m hicon.train --help."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import load_csv, make_datasets
from .engine import run_pipeline


def synthetic_demo(n=540, channels=3, seed=11):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    series = []
    for c in range(channels):
        y = (np.sin(t * (0.075 + 0.009*c) + c) +
             0.35*np.cos(t * (0.019 + 0.004*c)) +
             0.002*t + rng.normal(0, 0.045, n))
        series.append(y)
    return np.stack(series, axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="HiCon train/validate/test")
    parser.add_argument("--data", help="Chronological numeric CSV; optional date column")
    parser.add_argument("--columns", help="Comma-separated target feature names (defaults: all numeric)")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/default.json"))
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--backbone-epochs", type=int)
    parser.add_argument("--gate-epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default="runs/demo")
    parser.add_argument("--smoke", action="store_true", help="Quick artificial-data verification only")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    remap = {"lookback": "lookback", "horizon": "horizon",
             "batch_size": "batch_size", "backbone_epochs": "backbone_epochs",
             "gate_epochs": "gate_epochs", "seed": "seed"}
    for arg, key in remap.items():
        val = getattr(args, arg)
        if val is not None:
            cfg[key] = val
    if args.smoke:
        values = synthetic_demo()
        columns = [f"demo_{i}" for i in range(values.shape[1])]
        cfg.update(lookback=32, horizon=8, batch_size=24, d_model=16, d_ff=32,
                   backbone_epochs=2, gate_epochs=2, patience=2, seed=2021)
        source = "synthetic_demo"
    else:
        if not args.data:
            parser.error("--data is required unless --smoke is given")
        values, columns = load_csv(args.data, args.columns)
        source = str(Path(args.data).resolve())
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but no CUDA device is available")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    datasets, scaler, bounds = make_datasets(values, cfg["lookback"], cfg["horizon"],
         cfg["train_ratio"], cfg["val_ratio"])
    print(f"HiCon | device={device} | data={source} | n={len(values)} | channels={len(columns)}")
    print("windows train/val/test:", [len(x) for x in datasets], flush=True)
    beginning = time.time()
    metrics, states = run_pipeline(*datasets, scaler, cfg, device)
    metrics["runtime_seconds"] = round(time.time()-beginning, 2)
    metrics["data_source"] = source
    metrics["lookback"] = cfg["lookback"]
    metrics["horizon"] = cfg["horizon"]
    metrics["num_channels"] = len(columns)
    checkpoint = {"cfg": cfg, "columns": columns, "n_rows": len(values),
        "split_bounds": list(bounds),
        "scaler_mean": scaler[0].tolist(), "scaler_std": scaler[1].tolist(),
        **states, "alpha": float(metrics["alpha"]),
        "gate_enabled": bool(metrics["gate_enabled"])}
    torch.save(checkpoint, outdir / "hicon.pt")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("\nResults (test, original units):")
    for name in ("base", "population", "hicon"):
        x = metrics[name]
        print(f"{name:>12}: MSE={x['mse']:.6f} MAE={x['mae']:.6f} "
              f"mse_norm={x['mse_norm']:.6f} mae_norm={x['mae_norm']:.6f}")
    print(f"Saved: {outdir / 'hicon.pt'}", flush=True)


if __name__ == "__main__":
    main()
