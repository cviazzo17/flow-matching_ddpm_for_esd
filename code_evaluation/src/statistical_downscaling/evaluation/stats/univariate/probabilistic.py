# src/statistical_downscaling/evaluation/stats/univariate/probabilistic.py


import xarray as xr
from typing import Union
import numpy as np

def spread(preds: Union[xr.Dataset, xr.DataArray]) -> xr.Dataset:
    preds = preds.copy()
    if "member" not in preds.dims:
        raise Exception("The spread can only be computed for ensemble predictions.")
    
    m = preds.sizes["member"] # ensemble size
    ensemble_mean = preds.mean(dim="member")
    variance = ((preds - ensemble_mean)**2).sum(dim="member") / (m - 1)
    variance *= (m + 1) / m  # Variance correction: Fortin et al. (2014)
    return np.sqrt(variance.mean(dim="time"))

