# src/statistical_downscaling/evaluation/stats/univariate/marginals.py

import xarray as xr
import numpy as np
from collections.abc import Callable
from typing import Optional


def mean(da: xr.DataArray, dim: str="time") -> xr.DataArray:
    return da.mean(dim=dim, skipna=True)

def std(da: xr.DataArray, dim: str="time") -> xr.DataArray:
    return da.std(dim=dim, skipna=True)

def quantile(da: xr.DataArray, q: float, dim: str="time") -> xr.DataArray:
    return da.quantile(dim=dim, q=q, skipna=True)

def median(da: xr.DataArray, dim: str="time") -> xr.DataArray:
    return quantile(da=da, dim=dim, q=0.50)

def p95(da: xr.DataArray, dim: str="time") -> xr.DataArray:
    return quantile(da=da, dim=dim, q=0.95)

def p99(da: xr.DataArray, dim: str="time") -> xr.DataArray:
    return quantile(da=da, dim=dim, q=0.99)

# stats conditioned on exceedances above a threshold -------------------

def _get_wet_mask(da: xr.DataArray, threshold: float = 1.0) -> xr.DataArray:
    return (da >= threshold)

def _compute_wet_stat(da: xr.DataArray, stat_func: Callable, *, threshold: float = 1.0, dim: str="time", **kwargs) -> xr.DataArray:
    wet_mask = _get_wet_mask(da, threshold).compute()
    masked_da = da.where(wet_mask, drop=False)
    return stat_func(da=masked_da, dim=dim, **kwargs)

def wet_mean(da: xr.DataArray, threshold: float = 1.0, dim: str = "time") -> xr.DataArray:
    return _compute_wet_stat(da, mean, threshold=threshold, dim=dim)

def wet_std(da: xr.DataArray, threshold: float = 1.0, dim: str = "time") -> xr.DataArray:
    return _compute_wet_stat(da, std, threshold=threshold, dim=dim)

def wet_median(da: xr.DataArray, threshold: float = 1.0, dim: str = "time") -> xr.DataArray:
    return _compute_wet_stat(da, median, threshold=threshold, dim=dim)

def wet_p95(da: xr.DataArray, threshold: float = 1.0, dim: str = "time") -> xr.DataArray:
    return _compute_wet_stat(da, p95, threshold=threshold, dim=dim)

def wet_p99(da: xr.DataArray, threshold: float = 1.0, dim: str = "time") -> xr.DataArray:
    return _compute_wet_stat(da, p99, threshold=threshold, dim=dim)

def dry_days(da: xr.DataArray, threshold: float = 1.0, dim: str = "time") -> xr.DataArray:
    return (~(_get_wet_mask(da, threshold).compute())).sum(dim=dim)
             
def dry_days_percentage(da: xr.DataArray, threshold: float = 1.0, dim: str = "time") -> xr.DataArray:
    da = da.copy()
    dry_mask = (da < threshold).where(da.notnull())
    return dry_mask.mean(dim=dim, skipna=True) * 100

# --------------------------------------------------------------------------

def ecdf_1d(x: np.ndarray, pseudo: bool = False, 
         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Computes the ECDF as ranks / (T + 1). 
    
    If pseudo = True, ties are broken at random (e.g., dry days)."""
    
    if pseudo:
        T = x.size
        if rng is None:
            rng = np.random.default_rng(None)
        
        order = np.lexsort((rng.random(T), x))
        r = np.empty(T)
        r[order] = np.arange(1, T + 1)
        result = r / (T + 1)
    else:
        values, inverse, counts = np.unique(
            x, return_inverse=True, return_counts=True
        )

        cumulative = np.cumsum(counts) / x.size
        result = cumulative[inverse]
    return result


def ecdf_xa(
    da: xr.DataArray, pseudo: bool = False, 
    rng: Optional[np.random.Generator] = None,
    dim: str = "time"
) -> xr.DataArray:
    """Computes the ECDF as ranks / (T + 1) using the dimension `dim` 
    ("time" by default). 
    
    If pseudo = True, ties are broken at random (e.g., dry days)."""
    return xr.apply_ufunc(
        ecdf,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        kwargs={
           "rng": rng,
            "pseudo": pseudo,
        },
    )
