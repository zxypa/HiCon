"""Print HiCon's 44-setting normalized MSE and MAE; no external baselines."""
from pathlib import Path
import json
import argparse
import csv

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic", "ILI", "Covid-19", "NN5", "Wike2000"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/full44")
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    root = Path(args.runs)
    rows = []
    for dataset in DATASETS:
        horizons = (24, 36, 48, 60) if dataset in {"ILI", "Covid-19", "NN5", "Wike2000"} else (96, 192, 336, 720)
        for h in horizons:
            files = sorted(root.glob(f"{dataset}_H{h}_L*/metrics.json"))
            if not files:
                rows.append((dataset, h, "-", "-", "MISSING"))
                continue
            # Never average or silently cherry-pick multiple different L runs.
            if len(files) > 1:
                rows.append((dataset, h, "-", "-", "MULTIPLE L; inspect runs"))
                continue
            try:
                report = json.loads(files[0].read_text())
                mse, mae = report["hicon"]["mse_norm"], report["hicon"]["mae_norm"]
                rows.append((dataset, h, f"{mse:.6f}", f"{mae:.6f}", "OK"))
            except (OSError, KeyError, ValueError):
                rows.append((dataset, h, "-", "-", "FAILED"))
    print("\nHiCon: normalized MSE + MAE (44 settings; lower is better)\n")
    print(f"{'Dataset':<14} {'H':>4} {'MSE':>12} {'MAE':>12}  Status")
    print("-"*67)
    for d,h,mse,mae,status in rows:
        print(f"{d:<14} {h:>4} {mse:>12} {mae:>12}  {status}")
    print("-"*67)
    print(f"Valid results: {sum(r[4]=='OK' for r in rows)}/44")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(("Dataset", "H", "MSE_norm", "MAE_norm", "Status"))
            w.writerows(rows)
        print("CSV:", args.csv)


if __name__ == "__main__":
    main()
