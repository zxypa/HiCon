# HiCon: Hierarchical Consensus Learning for Non-Stationary Time Series Forecasting

PyTorch implementation of **HiCon**, a hierarchical consensus learning framework for non-stationary time series forecasting, combining **population-level consensus learning** and **instance-adaptive refinement** to improve forecasting accuracy and reliability.

## Method

HiCon leverages complementary temporal and spectral prediction views through a hierarchical consensus learning mechanism.

1. **Prediction View Learning:** Generate reference, temporal, and spectral forecasts to capture complementary temporal dynamics.
2. **Population-Level Consensus:** Learn a shared reconciliation profile from training-set prediction residuals and cross-view disagreement.
3. **Instance-Adaptive Refinement:** Introduce bounded, input-dependent adjustments to accommodate heterogeneous forecasting conditions.

The final prediction is obtained through hierarchical reconciliation:

`Y_final = Y_reference + (alpha * g[h] + delta(X,h)) * (Y_temporal - Y_spectral)`

Here, `g[h]` represents the population-level consensus profile, while `delta(X,h)` provides instance-adaptive refinement.

Consensus statistics are estimated using training data, and model selection is performed on validation data. No future observations or test statistics are used during inference.

## Requirements

Python 3.10+, PyTorch 2.2+, NumPy, Pandas. A CUDA GPU is recommended for benchmark experiments.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Synthetic end-to-end experiment
python -m hicon.train --smoke --out runs/smoke

# Generate an example dataset
python examples/make_demo.py

# Train HiCon
python -m hicon.train --data data/demo.csv --lookback 32 --horizon 8 \
  --batch-size 24 --backbone-epochs 2 --gate-epochs 2 --out runs/demo

# Evaluate the trained model
python -m hicon.evaluate --checkpoint runs/demo/hicon.pt --data data/demo.csv
```

## Datasets

HiCon supports experiments on widely used time series forecasting benchmarks:

- **ETT:** ETTh1, ETTh2, ETTm1, ETTm2
- **Weather**
- **Electricity**
- **Traffic**
- **ILI**
- **Covid-19**
- **NN5**
- **Wiki2000**

Place the corresponding CSV files in the `data/` directory. Numerical columns are used for multivariate forecasting, while an optional `date` column is excluded from prediction targets.

## Training and Evaluation

Example: ETTh1 with input length 96 and prediction horizon 96.

```bash
# Training
python -m hicon.train --data data/ETTh1.csv \
  --lookback 96 --horizon 96 --out runs/ETTh1_H96_L96

# Evaluation
python -m hicon.evaluate \
  --checkpoint runs/ETTh1_H96_L96/hicon.pt \
  --data data/ETTh1.csv
```

## Benchmark Experiments

The experiment pipeline supports **44 forecasting configurations** across multiple datasets and prediction horizons.

```bash
# Preview experiment configurations
python scripts/reproduce_all.py --data-dir data --out-dir runs/full44 --dry-run

# Run benchmark experiments
python scripts/reproduce_all.py --data-dir data --out-dir runs/full44

# Display MSE and MAE results
python scripts/show_results.py --runs runs/full44

# Export results to CSV
python scripts/show_results.py --runs runs/full44 --csv hicon_results.csv
```

The default input length for long-horizon forecasting is `L=96`. Consistent dataset partitions, preprocessing procedures, and evaluation protocols should be used for fair comparisons.

## Configuration

Model architecture, training parameters, population-level consensus learning, and instance-adaptive refinement settings are configured in:

`configs/default.json`

Common parameters can be overridden through command-line arguments. Experiment configurations and dataset split information are saved with the corresponding checkpoints.

## Metrics

Forecasting performance is evaluated using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.

- `mse`, `mae`: Evaluation metrics in the original data scale.
- `mse_norm`, `mae_norm`: Evaluation metrics in the training-standardized data space.

Lower values indicate better forecasting performance.

## Experimental Outputs

Each experiment saves the model checkpoint, configuration, and evaluation results.

```text
runs/
└── ETTh1_H96_L96/
    ├── hicon.pt
    ├── config.json
    └── metrics.json
```

The recorded results include reference forecasting performance, population-level consensus performance, complete HiCon performance, and the corresponding evaluation metrics.

## Reproducibility

HiCon provides scripts for model training, evaluation, and batch experimentation.

Random seeds, experimental configurations, and dataset information are recorded to facilitate reproducible experiments. Benchmark comparisons should follow consistent data preprocessing, input lengths, prediction horizons, and evaluation protocols.

## Tests

Run the available tests:

```bash
pip install pytest
python -m pytest -q tests
```

## Citation

If you find HiCon useful in your research, please consider citing:

**HiCon: Hierarchical Consensus Learning for Non-Stationary Time Series Forecasting**

Xiangyuan Zhu, Xin Peng, Xin Su, Xianhong Wen.

Citation information will be updated upon publication.

## License

This project is released under the MIT License. Third-party datasets and dependencies are subject to their respective licenses.
