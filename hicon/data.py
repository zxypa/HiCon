"""Chronological CSV loading, train-only feature scaling and leakage-free target splits."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_csv(path, columns=None):
    frame = pd.read_csv(path)
    if columns:
        names = [c.strip() for c in columns.split(",")]
        absent = [c for c in names if c not in frame.columns]
        if absent:
            raise ValueError(f"Unknown columns: {absent}")
        frame = frame[names]
    else:
        frame = frame.select_dtypes(include=[np.number])
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("No numeric series columns found")
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Series contains NaN/Inf; clean the CSV before training")
    return values, list(frame.columns)


class WindowDataset(Dataset):
    def __init__(self, values, lookback, horizon, target_begin, target_end):
        self.values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        # Every target window must fall wholly within its own split.
        self.start = max(int(target_begin), self.lookback)
        self.finish = int(target_end) - self.horizon + 1
        if self.finish <= self.start:
            raise ValueError("Split too short for lookback/horizon. Use more data or smaller horizon")

    def __len__(self):
        return self.finish - self.start

    def __getitem__(self, index):
        target_at = self.start + index
        return (self.values[target_at-self.lookback:target_at],
                self.values[target_at:target_at+self.horizon])


def make_datasets(values, lookback, horizon, train_ratio=0.7, val_ratio=0.1,
                  scaler=None, split_bounds=None):
    n = len(values)
    if split_bounds is None:
        t, v = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    else:
        t, v = map(int, split_bounds)
    if not (0 < t < v < n):
        raise ValueError("Invalid chronological split boundaries")
    if scaler is None:
        mu = values[:t].mean(axis=0, dtype=np.float64).astype(np.float32)
        sig = values[:t].std(axis=0, dtype=np.float64).astype(np.float32)
        sig = np.maximum(sig, 1e-5)
    else:
        mu, sig = scaler
        mu = np.asarray(mu, dtype=np.float32)
        sig = np.asarray(sig, dtype=np.float32)
        if len(mu) != values.shape[1] or len(sig) != values.shape[1]:
            raise ValueError("Checkpoint feature count differs from CSV")
    data = ((values - mu) / sig).astype(np.float32)
    train = WindowDataset(data, lookback, horizon, 0, t)
    val = WindowDataset(data, lookback, horizon, t, v)
    test = WindowDataset(data, lookback, horizon, v, n)
    return (train, val, test), (mu, sig), (t, v)
