# src/statistical_downscaling/evaluation/metrics/univariate/temporal.py

import numpy as np
import xarray as xr
from collections.abc import Callable
import xskillscore


import numpy as np
import xarray as xr
from scipy import stats


def corr(target: xr.DataArray, pred: xr.DataArray,
         corr_type: str, deseasonal: bool) -> xr.DataArray:

    if corr_type not in ("spearman", "pearson"):
        raise ValueError("Provide a valid value for corr_type (spearman or pearson)")

    target = target.assign_coords(time=target.time.dt.floor("D"))
    pred = pred.assign_coords(time=pred.time.dt.floor("D"))
    n_t, n_p = target.sizes["time"], pred.sizes["time"]
    target, pred = xr.align(target, pred, join="inner")
    if target.sizes["time"] < max(n_t, n_p):
        print(f"[corr] using {target.sizes['time']} common days (target: {n_t}, pred: {n_p})")

    if deseasonal:
        target = target.load()
        target = (target.groupby("time.month") - target.groupby("time.month").mean("time")).drop_vars("month")
        pred = pred.load()
        pred = (pred.groupby("time.month") - pred.groupby("time.month").mean("time")).drop_vars("month")

    valid = target.notnull() & pred.notnull()
    target, pred = target.where(valid), pred.where(valid)

    if corr_type == "spearman":
        rank = lambda da: xr.apply_ufunc(
            lambda x: stats.rankdata(x, axis=-1, nan_policy="omit"),
            da, input_core_dims=[["time"]], output_core_dims=[["time"]],
            dask="parallelized", output_dtypes=[float],
        )
        target, pred = rank(target), rank(pred)

    return xr.corr(target, pred, dim="time").rename(f"{corr_type}_r")

