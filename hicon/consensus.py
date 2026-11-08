"""Population-level consensus and conservative validation-scale selection."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import corrected_prediction


@torch.no_grad()
def solve_population_profile(backbone, loader: DataLoader, device, ridge=0.01,
                             smooth=0.1, clip=0.2):
    """Solve (Gamma + ridge*I + smooth*L)g = c from training windows only."""
    backbone.eval()
    energy = None
    align = None
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        views = backbone(x)
        d = (views["temporal"] - views["spectral"]).double()
        e = (y - views["reference"]).double()
        g_batch = d.square().sum(dim=(0, 2)).cpu()
        c_batch = (e * d).sum(dim=(0, 2)).cpu()
        energy = g_batch if energy is None else energy + g_batch
        align = c_batch if align is None else align + c_batch
        n += len(x)
    if n == 0:
        raise ValueError("Empty training set")
    gamma = energy / n
    corr = align / n
    h = gamma.numel()
    lap = torch.zeros(h, h, dtype=torch.float64)
    if h > 1:
        i = torch.arange(h-1)
        lap[i, i] += 1
        lap[i+1, i+1] += 1
        lap[i, i+1] -= 1
        lap[i+1, i] -= 1
    matrix = torch.diag(gamma) + float(ridge) * torch.eye(h, dtype=torch.float64) + float(smooth) * lap
    profile = torch.linalg.solve(matrix, corr)
    return profile.clamp(-float(clip), float(clip)).float()


@torch.no_grad()
def choose_global_scale(backbone, loader, profile, device,
                        candidates=(0.0, 0.25, 0.5, 1.0), min_mean=0.0005,
                        min_median=0.0, min_fraction=0.5, near_best_tol=0.0005):
    """Gains are relative PER-WINDOW MSE reductions; 0.0005 = 0.05%."""
    backbone.eval()
    store = {float(a): [] for a in candidates}
    if 0.0 not in store:
        store[0.0] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        views = backbone(x)
        baseline = (views["reference"] - y).square().mean(dim=(1, 2)).clamp_min(1e-9)
        for alpha in store:
            correction = corrected_prediction(views, profile, alpha=alpha)
            score = (correction-y).square().mean(dim=(1, 2))
            store[alpha].append(((baseline-score) / baseline).cpu().numpy())
    stats = {}
    feasible = []
    for alpha, chunks in store.items():
        gains = np.concatenate(chunks)
        mean, med, fraction = float(gains.mean()), float(np.median(gains)), float((gains > 0).mean())
        stats[str(alpha)] = {"mean_gain": mean, "median_gain": med, "positive_fraction": fraction}
        if alpha != 0.0 and mean >= min_mean and med >= min_median and fraction >= min_fraction:
            feasible.append(alpha)
    if not feasible:
        return 0.0, stats
    best = max(stats[str(a)]["mean_gain"] for a in feasible)
    accepted = [a for a in feasible if stats[str(a)]["mean_gain"] >= best - near_best_tol]
    return min(accepted, key=lambda a: (abs(a), a)), stats
