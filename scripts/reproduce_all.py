"""Run 44 dataset/horizon settings, Electricity and Traffic LAST. Skips missing CSVs."""
from pathlib import Path
import argparse
import subprocess
import sys

LONG = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
SHORT = ["ILI", "Covid-19", "NN5", "Wike2000"]
LARGE = ["Electricity", "Traffic"]


def plan():
    return ([(d, h, 96) for d in LONG for h in (96, 192, 336, 720)] +
            [(d, h, 36 if d == "Covid-19" else 104) for d in SHORT for h in (24, 36, 48, 60)] +
            [(d, h, 96) for d in LARGE for h in (96, 192, 336, 720)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="runs/full44")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--lookback-long", type=int, default=96,
                    help="Use 512 only to explore the historical setting; it is NOT a matched L=96 comparison")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repo = Path(args.data_dir)
    for i, (dataset, horizon, lookback) in enumerate(plan(), 1):
        src = repo / f"{dataset}.csv"
        outdir = Path(args.out_dir) / f"{dataset}_H{horizon}_L{args.lookback_long if lookback == 96 else lookback}"
        print(f"[{i}/44] {dataset} H={horizon}", flush=True)
        if not src.exists():
            print(f"[SKIP] missing {src}", flush=True)
            continue
        if (outdir / "metrics.json").exists() and (outdir / "hicon.pt").exists():
            print("[SKIP] completed", flush=True)
            continue
        # Keep GPU memory pressure manageable for high-channel benchmarks.
        if dataset == "Traffic":
            batch_size = 2 if horizon <= 192 else 1
        elif dataset == "Electricity":
            batch_size = 8 if horizon <= 192 else 4
        else:
            batch_size = 32
        cmd = [sys.executable, "-m", "hicon.train", "--data", str(src),
               "--horizon", str(horizon), "--lookback", str(args.lookback_long if lookback == 96 else lookback),
               "--batch-size", str(batch_size), "--device", args.device, "--out", str(outdir)]
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            code = subprocess.call(cmd)
            if code:
                print(f"[FAILED] exit code {code}; continue to next setting", flush=True)

if __name__ == "__main__":
    main()
