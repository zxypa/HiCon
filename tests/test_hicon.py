import numpy as np
import torch
from hicon.model import TemporalSpectralForecaster, InstanceRefinement, corrected_prediction
from hicon.consensus import solve_population_profile, choose_global_scale
from hicon.data import make_datasets
from hicon.engine import loader


def test_views_and_gate_shapes():
    torch.manual_seed(1)
    model = TemporalSpectralForecaster(32, 8, d_model=16, d_ff=32, n_heads=2, n_experts=2, patch_len=8)
    x = torch.randn(3, 32, 2)
    v = model(x)
    for key in ("reference", "temporal", "spectral"):
        assert v[key].shape == (3, 8, 2)
    gate = InstanceRefinement(16, 16, 0.05)
    delta = gate(x, v)
    assert delta.shape == (3, 8)
    assert delta.abs().max() <= 0.05 + 1e-6
    p = corrected_prediction(v, torch.zeros(8), 1.0, delta)
    assert p.shape == (3, 8, 2)


def test_population_profile():
    values = np.sin(np.arange(440, dtype=np.float32) / 9).reshape(-1, 1)
    (tr, va, _), _, _ = make_datasets(values, 32, 8)
    net = TemporalSpectralForecaster(32, 8, d_model=16, d_ff=32, dropout=0.0)
    profile = solve_population_profile(net, loader(tr, 32), "cpu")
    assert profile.shape == (8,)
    assert torch.isfinite(profile).all()
    assert profile.abs().max() <= .200001
    alpha, stats = choose_global_scale(net, loader(va, 32), profile, "cpu")
    assert alpha in (0., .25, .5, 1.)
    assert "0.0" in stats
