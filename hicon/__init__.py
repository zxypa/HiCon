"""HiCon: Hierarchical Consensus Learning for time-series forecasting."""
from .model import TemporalSpectralForecaster, InstanceRefinement
from .consensus import solve_population_profile, choose_global_scale
__all__ = ["TemporalSpectralForecaster", "InstanceRefinement", "solve_population_profile", "choose_global_scale"]
