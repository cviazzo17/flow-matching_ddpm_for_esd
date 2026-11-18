# src/statistical_downscaling/evaluation/metrics/univariate/marginals.py

import numpy as np
import xarray as xr
from collections.abc import Callable
from ...stats.univariate.marginals import mean, std, p99, wet_mean, wet_std, wet_p99, dry_days_percentage


# helpers
def _compute_bias(pred: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    return pred - target


def _compute_ratio(pred: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    return pred / target


def _compute_rel_error(pred: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    return (pred - target) / target


def _compute_ratio_for_stat(
    preds: xr.DataArray,
    target: xr.DataArray,
    stat_func: Callable,
    dim: str = "time",
    **kwargs,
) -> xr.DataArray:
    pred_stat = stat_func(da=preds, dim=dim, **kwargs)
    target_stat = stat_func(da=target, dim=dim, **kwargs)
    return _compute_ratio(pred=pred_stat, target=target_stat)


def _compute_bias_for_stat(
    preds: xr.DataArray,
    target: xr.DataArray,
    stat_func: Callable,
    dim: str = "time",
    **kwargs,
) -> xr.DataArray:
    pred_stat = stat_func(da=preds, dim=dim, **kwargs)
    target_stat = stat_func(da=target, dim=dim, **kwargs)
    return _compute_bias(pred=pred_stat, target=target_stat)


def _compute_rel_error_for_stat(
    preds: xr.DataArray,
    target: xr.DataArray,
    stat_func: Callable,
    dim: str = "time",
    **kwargs,
) -> xr.DataArray:
    pred_stat = stat_func(da=preds, dim=dim, **kwargs)
    target_stat = stat_func(da=target, dim=dim, **kwargs)
    return _compute_rel_error(pred=pred_stat, target=target_stat)

# -------------


def rel_error_mean(
    preds: xr.DataArray, target: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    return _compute_rel_error_for_stat(preds, target, mean, dim=dim)


def rel_error_std(
    preds: xr.DataArray, target: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    return _compute_rel_error_for_stat(preds, target, std, dim=dim)


def rel_error_p95(
    preds: xr.DataArray, target: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    return _compute_rel_error_for_stat(preds, target, p95, dim=dim)


def rel_error_p99(
    preds: xr.DataArray, target: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    return _compute_rel_error_for_stat(preds, target, p99, dim=dim)


def rel_error_wet_mean(    
    preds: xr.DataArray, target: xr.DataArray, threshold: float = 1.0, 
    dim: str = "time"
) -> xr.DataArray:
    return _compute_rel_error_for_stat(preds, target, wet_mean, dim=dim, threshold=threshold)


def rel_error_wet_std(    
    preds: xr.DataArray, target: xr.DataArray, threshold: float = 1.0, 
    dim: str = "time"
) -> xr.DataArray:
    return _compute_rel_error_for_stat(preds, target, wet_std, dim=dim, threshold=threshold)


def rel_error_wet_p95(    
    preds: xr.DataArray, target: xr.DataArray, threshold: float = 1.0, 
    dim: str = "time"
) -> xr.DataArray:
    return _compute_rel_error_for_stat(preds, target, wet_p95, dim=dim, threshold=threshold)


def rel_error_wet_p99(    
    preds: xr.DataArray, target: xr.DataArray, threshold: float = 1.0, 
    dim: str = "time"
) -> xr.DataArray:
    return _compute_rel_error_for_stat(preds, target, wet_p99, dim=dim, threshold=threshold)


def rel_error_dry_days(
    preds: xr.DataArray, target: xr.DataArray, threshold: float = 1.0, 
    dim: str = "time",
):
    return _compute_rel_error_for_stat(preds, target, dry_days_percentage, dim=dim, threshold=threshold)


# --------------------------------------------------------

def pit_1d(ref: np.ndarray, x: np.ndarray):
    """Conducts the one-dimensional Probability Integral Tranform over x
    using a reference ref.
    ECDF_ref(x).
    """
    r = np.sort(ref[np.isfinite(ref)])
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if r.size:
        out[ok] = np.searchsorted(r, x[ok], side="right") / (r.size + 1)
    return out


def pit_xa(preds: xr.DataArray, target: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Conducts the n-dimensional Probability Integral Tranform over x
    using a reference ref.
    By default, the PIT is computed across the "time" dimension.
    """
    return xr.apply_ufunc(
        pit_1d,
        target.rename({dim: "_tr"}),
        preds,
        input_core_dims=[["_tr"], [dim]],
        output_core_dims=[[dim]],
        vectorize=True,
    ).transpose(*preds.dims)
    
