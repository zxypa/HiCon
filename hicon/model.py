"""Forecast views and bounded instance-adaptive correction.

Input:  [batch, lookback, channels]
Output: three [batch, horizon, channels] predictions in the SAME normalized space.
This is a standalone implementation, not an exact copy of the authors' historical code.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalSpectralForecaster(nn.Module):
    def __init__(self, lookback: int, horizon: int, d_model: int = 32,
                 d_ff: int = 512, n_heads: int = 2, n_experts: int = 2,
                 patch_len: int = 8, dropout: float = 0.1):
        super().__init__()
        if lookback < 2 or horizon < 1 or patch_len < 1 or n_experts < 1:
            raise ValueError("Lookback >=2, horizon >=1, patch_len and n_experts >=1 required")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model
        self.patch_len = patch_len
        self.n_patches = math.ceil(lookback / patch_len)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=False
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.temporal_experts = nn.ModuleList([
            nn.Linear(self.n_patches * d_model, horizon) for _ in range(n_experts)
        ])
        self.expert_router = nn.Linear(d_model, n_experts)
        fft_bins = lookback // 2 + 1
        self.spectral_head = nn.Sequential(
            nn.Linear(2 * fft_bins, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, horizon)
        )
        self.reference_mix = nn.Parameter(torch.zeros(horizon))

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[1] != self.lookback:
            raise ValueError(f"Expected [B,{self.lookback},C], got {tuple(x.shape)}")
        b, _, c = x.shape
        # Reversible per-window normalization. Differentiable in x but x has no trainable params.
        mu = x.mean(dim=1, keepdim=True).detach()
        sig = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        x_norm = (x - mu) / sig
        patches = x_norm.transpose(1, 2)  # [B,C,L]
        pad = self.n_patches * self.patch_len - self.lookback
        if pad:
            patches = F.pad(patches, (0, pad), mode="replicate")
        tokens = patches.reshape(b * c, self.n_patches, self.patch_len)
        tokens = self.patch_embed(tokens) + self.position
        hidden = self.temporal_encoder(tokens)  # [B*C,P,D]
        pool = hidden.mean(dim=1)
        expert_weights = self.expert_router(pool).softmax(dim=-1)
        expert_outputs = torch.stack(
            [head(hidden.flatten(1)) for head in self.temporal_experts], dim=1
        )  # [B*C,E,H]
        temporal = (expert_outputs * expert_weights.unsqueeze(-1)).sum(dim=1)
        temporal = temporal.reshape(b, c, self.horizon).transpose(1, 2)

        freq = torch.fft.rfft(x_norm.transpose(1, 2), dim=-1, norm="ortho")
        freq_features = torch.cat([freq.real, freq.imag], dim=-1)
        spectral = self.spectral_head(freq_features).transpose(1, 2)
        mix = torch.sigmoid(self.reference_mix)[None, :, None]
        reference = mix * temporal + (1.0 - mix) * spectral
        # Restore per-instance scale for all outputs so disagreement and residual agree.
        reference = reference * sig + mu
        temporal = temporal * sig + mu
        spectral = spectral * sig + mu
        context = pool.reshape(b, c, self.d_model).mean(dim=1)
        return {"reference": reference, "temporal": temporal,
                "spectral": spectral, "context": context}


class InstanceRefinement(nn.Module):
    """Per-horizon shared MLP, using ONLY observed inputs and frozen-model predictions."""
    def __init__(self, context_dim: int, hidden_dim: int = 16, epsilon: float = 0.05):
        super().__init__()
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        self.epsilon = float(epsilon)
        # 6 input summaries + 4 disagreement/forecast summaries + horizon position.
        self.network = nn.Sequential(
            nn.Linear(context_dim + 11, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, history: torch.Tensor, views: dict[str, torch.Tensor]):
        reference = views["reference"]
        disagreement = views["temporal"] - views["spectral"]
        b, h, _ = disagreement.shape
        scale = disagreement.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-4)
        d = disagreement / scale
        reference_scale = reference.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-4)
        horizon_stats = torch.stack([
            d.mean(dim=-1), d.abs().mean(dim=-1),
            d.square().mean(dim=-1).sqrt(), (reference / reference_scale).mean(dim=-1)
        ], dim=-1)  # [B,H,4]
        first_diff = history[:, 1:, :] - history[:, :-1, :]
        spectrum = torch.fft.rfft(history, dim=1, norm="ortho").abs()
        cutoff = max(1, spectrum.shape[1] // 2)
        high_energy = spectrum[:, cutoff:, :].square().sum(dim=(1, 2))
        all_energy = spectrum.square().sum(dim=(1, 2)).clamp_min(1e-6)
        high_ratio = high_energy / all_energy
        global_stats = torch.stack([
            history.mean(dim=(1, 2)),
            history.std(dim=(1, 2), unbiased=False),
            (history[:, -1, :] - history[:, 0, :]).mean(dim=-1),
            first_diff.abs().mean(dim=(1, 2)),
            first_diff.std(dim=(1, 2), unbiased=False),
            high_ratio,
        ], dim=-1)
        global_stats = torch.nan_to_num(global_stats, nan=0.0, posinf=0.0, neginf=0.0)
        context = views["context"]
        position = torch.linspace(0, 1, h, device=history.device, dtype=history.dtype)[None, :, None]
        features = torch.cat([
            context[:, None, :].expand(-1, h, -1),
            global_stats[:, None, :].expand(-1, h, -1),
            horizon_stats,
            position.expand(b, -1, -1)
        ], dim=-1)
        return self.epsilon * torch.tanh(self.network(features).squeeze(-1))  # [B,H]


def corrected_prediction(views, profile, alpha=0.0, residual=None):
    d = views["temporal"] - views["spectral"]
    coefficients = (float(alpha) * profile).to(d.device, d.dtype)[None, :]
    if residual is not None:
        coefficients = coefficients + residual
    return views["reference"] + coefficients[:, :, None] * d
