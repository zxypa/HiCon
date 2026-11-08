# HiCon: Hierarchical Consensus Learning for Non-Stationary Time Series Forecasting

Research-oriented PyTorch implementation of a hierarchical prediction-level reconciliation mechanism combining **population consensus** and **bounded instance-adaptive refinement**.

> **Implementation provenance:** This repository is a **new, standalone reconstruction** of the HiCon method from the mathematical description and previously recorded configuration notes. It is **not byte-for-byte identical to the historical `SHDR-v1`/HiCon code** used for the manuscript tables. In particular, the temporal/spectral view-generation backbone is self-contained here rather than a verbatim reproduction of that earlier backbone. Benchmark scores from the manuscript are **not claimed to be reproduced** by this repository until independently verified. No external code is copied into this release.

## Method

For input `X`, a common temporal–spectral forecaster returns reference, temporal and spectral predictions. The disagreement `D = Y_temporal - Y_spectral` determines a constrained correction direction:

`Y_final = Y_reference + (alpha * g[h] + delta(X,h)) * D`.

1. **Backbone:** train three outputs jointly, then freeze the network.
2. **Population profile:** compute training-set residual–disagreement statistics, solve `(Gamma + lambda_r I + lambda_s L)g = c`, and select `alpha` using validation gains, medians, and positive-window rates.
3. **Instance refinement:** learn a small horizon-wise MLP with `delta = epsilon * tanh(MLP(...))`, retaining it only after improving validation MSE.

No future labels or test statistics are used for estimating corrections or model selection. The gate's observable descriptors include historical statistics, a frozen encoder representation, forecast disagreement and normalized horizon position.

## Requirements

Python 3.10+, PyTorch 2.2+, NumPy, Pandas. CPU works for a smoke test; a CUDA GPU is recommended for real benchmarks.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Quick start

```bash
# Synthetic end-to-end train -> population consensus -> refinement -> test
python -m hicon.train --smoke --out runs/smoke

# Or prepare an example CSV first
python examples/make_demo.py
python -m hicon.train --data data/demo.csv --lookback 32 --horizon 8 \
  --batch-size 24 --backbone-epochs 2 --gate-epochs 2 --out runs/demo
python -m hicon.evaluate --checkpoint runs/demo/hicon.pt --data data/demo.csv
```

## Real datasets

Put CSV files at `data/ETTh1.csv`, etc. A `date` column is ignored; numeric columns are forecast jointly.

```bash
python -m hicon.train --data data/ETTh1.csv \
  --lookback 96 --horizon 96 --out runs/ETTh1_H96_L96
python -m hicon.evaluate --checkpoint runs/ETTh1_H96_L96/hicon.pt --data data/ETTh1.csv
```

## All 44 planned settings

`Electricity` and `Traffic` run last. Missing CSVs are skipped. Results from distinct look-back lengths are never silently merged.

```bash
python scripts/reproduce_all.py --data-dir data --out-dir runs/full44 --dry-run
python scripts/reproduce_all.py --data-dir data --out-dir runs/full44
python scripts/show_results.py --runs runs/full44
python scripts/show_results.py --runs runs/full44 --csv hicon_results.csv
```

The default long-horizon look-back is `L=96`. Historical SHDR-v1 experiments used other look-backs (including `L=512`), so **do not relabel old numerical results as `L=96`**. The generic 70/10/20 chronological splits in this standalone implementation **do not automatically replicate TFB rolling evaluation**; use the published split/evaluation implementation for strict fair comparisons.

## Configuration

Change hyperparameters in `configs/default.json`; CLI flags override common training sizes. Configuration defaults are motivated by the HiCon design notes, but are **not a verified exact snapshot** of historical runs. Full configuration and split metadata are saved with every checkpoint.

## Metrics

- `mse`, `mae`: original data units (inverse global training-set standardization).
- `mse_norm`, `mae_norm`: metrics in the **train-standardized data space**; this is **not necessarily identical** to any external benchmark's `mse_norm` definition.

## Reproducibility and citation

A run saves `hicon.pt`, `config.json`, `metrics.json`, including backbone/base, population-only and complete HiCon scores plus selected alpha and whether the gate is enabled. Dataset source and seed are recorded. Consider multiple seeds and report uncertainty before making robustness claims.

A manuscript citation entry may be added after publication. Do not claim the model is backbone-independent without running additional backbone experiments.

## License

MIT license applies only to code newly written in this standalone repository. Datasets and third-party packages retain their respective licenses.

## Sanity checks

```bash
pip install pytest
python -m pytest -q tests
```

## Notes for maintainers

- Never publish historical manuscript MSE/MAE as if generated from this reconstructed version.
- `scripts/reproduce_all.py` uses a generic loader and will not automatically reproduce the official TFB test protocol.
- For final paper reproducibility, replace the independent backbone here with the archived model implementation and verify the official dataset splits, input lengths, selection logic, and checkpoints.
