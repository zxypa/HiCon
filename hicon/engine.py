"""Train backbone, estimate population profile, train bounded gate, rollback on validation."""
from __future__ import annotations
import copy
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import TemporalSpectralForecaster, InstanceRefinement, corrected_prediction
from .consensus import solve_population_profile, choose_global_scale


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=False)


@torch.no_grad()
def validate_backbone(model, val_loader, device):
    model.eval()
    total, count = 0.0, 0
    for x, y in val_loader:
        views = model(x.to(device))
        total += (views["reference"]-y.to(device)).abs().sum().item()
        count += y.numel()
    return total / count


def train_backbone(model, train_loader, val_loader, cfg, device):
    optim = torch.optim.Adam(model.parameters(), lr=float(cfg["backbone_lr"]))
    best, state, bad = float("inf"), None, 0
    for epoch in range(int(cfg["backbone_epochs"])):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            loss = (model(x)["reference"] - y).abs().mean()
            loss.backward()
            optim.step()
        val = validate_backbone(model, val_loader, device)
        print(f"[A] epoch={epoch+1:03d} val_mae={val:.6f}", flush=True)
        if val < best - 1e-6:
            best, state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    if state is not None:
        model.load_state_dict(state)
    return best


@torch.no_grad()
def evaluate_model(backbone, loader_, device, profile=None, alpha=0.0, gate=None, scaler=None):
    backbone.eval()
    if gate is not None:
        gate.eval()
    sq, ab, sq_norm, ab_norm, n = 0., 0., 0., 0., 0
    for x, y in loader_:
        x, y = x.to(device), y.to(device)
        views = backbone(x)
        if profile is None:
            pred = views["reference"]
        else:
            delta = gate(x, views) if gate is not None else None
            pred = corrected_prediction(views, profile, alpha, delta)
        diff = pred - y
        sq_norm += diff.square().sum().item()
        ab_norm += diff.abs().sum().item()
        if scaler is not None:
            _, sig = scaler
            s = torch.as_tensor(sig, device=device, dtype=diff.dtype)[None, None, :]
            diff = diff * s
        sq += diff.square().sum().item()
        ab += diff.abs().sum().item()
        n += diff.numel()
    return {"mse": sq/n, "mae": ab/n, "mse_norm": sq_norm/n, "mae_norm": ab_norm/n}


@torch.no_grad()
def validate_gate(backbone, gate, data_loader, device, profile, alpha):
    backbone.eval()
    gate.eval()
    baseline_loss, gate_loss, count = 0.0, 0.0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        views = backbone(x)
        population = corrected_prediction(views, profile, alpha)
        full = corrected_prediction(views, profile, alpha, gate(x, views))
        baseline_loss += (population-y).square().sum().item()
        gate_loss += (full-y).square().sum().item()
        count += y.numel()
    return baseline_loss/count, gate_loss/count


def train_gate(backbone, gate, train_loader, val_loader, profile, alpha, cfg, device):
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(gate.parameters(), lr=float(cfg["gate_lr"]))
    best, selected, stale = float("inf"), None, 0
    for epoch in range(int(cfg["gate_epochs"])):
        gate.train()
        total, steps = 0., 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                views = backbone(x)
            delta = gate(x, views)
            pred = corrected_prediction(views, profile, alpha, delta)
            loss = (pred-y).square().mean() + float(cfg["gate_regularizer"]) * delta.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
        pop_val, gate_val = validate_gate(backbone, gate, val_loader, device, profile, alpha)
        print(f"[C] epoch={epoch+1:03d} loss={total/max(steps,1):.6f} val_mse={gate_val:.6f}", flush=True)
        if gate_val < best - 1e-8:
            best, selected, stale = gate_val, copy.deepcopy(gate.state_dict()), 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if selected is not None:
        gate.load_state_dict(selected)
    pop_val, gate_val = validate_gate(backbone, gate, val_loader, device, profile, alpha)
    gain = (pop_val - gate_val) / max(pop_val, 1e-9)
    accepted = gain >= float(cfg["gate_min_gain"])
    print(f"[C] validation relative gain={gain:.3%}; gate={'ON' if accepted else 'OFF'}", flush=True)
    return accepted, gain


def run_pipeline(train_set, val_set, test_set, scaler, cfg, device):
    seed_everything(int(cfg["seed"]))
    net_cfg = {k: cfg[k] for k in ["lookback", "horizon", "d_model", "d_ff", "n_heads", "n_experts", "patch_len", "dropout"]}
    backbone = TemporalSpectralForecaster(**net_cfg).to(device)
    tr = loader(train_set, cfg["batch_size"], shuffle=True)
    tr_fixed = loader(train_set, cfg["batch_size"])
    va = loader(val_set, cfg["batch_size"])
    te = loader(test_set, cfg["batch_size"])

    print("Stage A: train reference forecaster", flush=True)
    train_backbone(backbone, tr, va, cfg, device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    base = evaluate_model(backbone, te, device, scaler=scaler)
    print("Stage B: solve population consensus", flush=True)
    profile = solve_population_profile(backbone, tr_fixed, device, cfg["profile_ridge"],
                                       cfg["profile_smooth"], cfg["profile_clip"])
    alpha, alpha_stats = choose_global_scale(backbone, va, profile, device,
        cfg["alpha_candidates"], cfg["selection_min_mean"], cfg["selection_min_median"],
        cfg["selection_min_fraction"], cfg["selection_near_best_tol"])
    population = evaluate_model(backbone, te, device, profile=profile, alpha=alpha, scaler=scaler)
    print(f"[B] selected alpha={alpha}, population test MSE={population['mse']:.6f}", flush=True)
    gate = InstanceRefinement(cfg["d_model"], cfg["gate_hidden"], cfg["epsilon"]).to(device)
    print("Stage C: train bounded instance refinement", flush=True)
    gate_on, gate_gain = train_gate(backbone, gate, tr, va, profile, alpha, cfg, device)
    final = evaluate_model(backbone, te, device, profile=profile, alpha=alpha,
                           gate=gate if gate_on else None, scaler=scaler)
    metrics = {"base": base, "population": population, "hicon": final,
               "alpha": alpha, "gate_enabled": gate_on,
               "gate_validation_relative_gain": gate_gain, "alpha_candidates": alpha_stats}
    model_states = {"backbone": backbone.state_dict(), "gate": gate.state_dict(),
                    "profile": profile.detach().cpu()}
    return metrics, model_states
