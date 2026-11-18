# src/statistical_downscaling/evaluation/metrics/univariate/probabilistic.py
from typing import Union 
import numpy as np 
import xarray as xr
from ...stats.univariate.probabilistic import spread


def spread_skill(preds: Union[xr.Dataset, xr.DataArray], target: Union[xr.Dataset, xr.DataArray]) -> xr.Dataset:
    if "member" not in preds.dims:
        raise ValueError("The spread can only be computed for ensemble predictions.")
    ens_spread = spread(preds)
    rmse = np.sqrt(((target - preds.mean(dim="member"))**2).mean(dim="time"))
    if isinstance(rmse, xr.DataArray):
        name = rmse.name or "variable"
        ens_spread = ens_spread.rename(name) if isinstance(ens_spread, xr.DataArray) else ens_spread
        return xr.Dataset({"spread_skill": ens_spread / rmse, "spread": ens_spread, "rmse": rmse})
    var_ds = list(rmse.data_vars)[0]
    return xr.merge([(ens_spread / rmse).rename({var_ds: "spread_skill"}), ens_spread.rename({var_ds: "spread"}), rmse.rename({var_ds: "rmse"})])


def crps_ensemble(preds: xr.Dataset, target: xr.Dataset, time_chunk: int = 50) -> xr.Dataset:
    if "member" not in preds.dims:
        raise Exception("CRPS requires ensemble predictions.")
    
    crps_chunks = []
    times = preds["time"].values
    for i in range(0, len(times), time_chunk):
        t_slice = times[i:i+time_chunk]
        p = preds.sel(time=t_slice)
        o = target.sel(time=t_slice)
        
        term1 = np.abs(p - o).mean(dim="member")
        p_pair = p.rename(member="member2")
        term2 = np.abs(p - p_pair).mean(dim=["member", "member2"])
        
        crps_chunks.append((term1 - 0.5 * term2))
    
    return xr.concat(crps_chunks, dim="time").mean(dim="time")


def crpss(crps_model: xr.Dataset, crps_ref: xr.Dataset) -> xr.Dataset:
    return 1 - crps_model / crps_ref


def climatological_reference(target: xr.Dataset, window_days: int = 14) -> xr.Dataset:
    doy = target["time"].dt.dayofyear.values
    years = target["time"].dt.year.values
    n_years_total = len(np.unique(years))
    
    clim_by_doy = {}
    for d in np.unique(doy):
        doy_diff = np.minimum(np.abs(doy - d), 365 - np.abs(doy - d))
        mask = doy_diff <= window_days
        
        windowed = target.isel(time=mask)
        windowed = windowed.assign_coords(year=("time", years[mask]))
        clim_by_doy[d] = windowed.groupby("year").mean(dim="time")
    
    clim_samples = []
    for i, t in enumerate(target["time"].values):
        d, y0 = doy[i], years[i]
        
        pool = clim_by_doy[d]
        mask_year = pool["year"].values != y0
        members = pool.isel(year=mask_year)
        
        expected = n_years_total - 1
        if members.sizes["year"] != expected:
            raise ValueError(
                f"time={t}: esperaba {expected} members, obtuve {members.sizes['year']}"
            )
        
        members = members.rename(year="member")
        members = members.assign_coords(member=np.arange(members.sizes["member"]))
        clim_samples.append(members.expand_dims(time=[t]))
    
    return xr.concat(clim_samples, dim="time")
