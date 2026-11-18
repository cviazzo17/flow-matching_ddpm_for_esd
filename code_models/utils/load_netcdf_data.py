"""Functions for loading and preparing the data to train and test the 
deep learning models."""

from typing import Optional

import xarray as xr
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import os


def _sin_doy(dates, leap_years=False):
    """
    Returns the sine of the the doy/count_of_doys_in_year.

    doy stands out for day of the year.
    """
    dates = pd.to_datetime(dates)
    doy = dates.dayofyear
    if leap_years:
        days_in_year = np.where(dates.is_leap_year, 366, 365)
    else:
        days_in_year = 365
    return np.sin(2 * np.pi * (doy / days_in_year))


def _cos_doy(dates, leap_years=False):
    """
    Returns the consine of the the doy/count_of_doys_in_year.

    doy stands out for day of the year.
    """
    dates = pd.to_datetime(dates)
    doy = dates.dayofyear
    if leap_years:
        days_in_year = np.where(dates.is_leap_year, 366, 365)
    else:
        days_in_year = 365
    return np.cos(2 * np.pi * (doy / days_in_year))


def get_x1d(x2d: xr.Dataset, add_seasonal_predictors: bool = True) -> xr.DataArray:
    """
    Returns the one-dimensional predictors, derived from the two-dimensional
    predictors and the day of the year (see Doury et al., 2024).
    """
    stds = x2d.std(dim=("lat", "lon")).compute()
    means = x2d.mean(dim=("lat", "lon")).compute()
    means = means.rename_vars({var_name: f"{var_name}_mean" for var_name in means.data_vars})
    stds = stds.rename_vars({var_name: f"{var_name}_std" for var_name in stds.data_vars})
    x1d = xr.merge([means, stds])
    if add_seasonal_predictors:
        x1d['sin_doy'] = xr.apply_ufunc(_sin_doy, x2d.time)
        x1d['cos_doy'] = xr.apply_ufunc(_cos_doy, x2d.time)
    return x1d


def align_datasets(
    data_1: xr.Dataset, data_2: xr.Dataset, coord: str
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Align two Datasets with respect to the coord.
    """
    data_1 = data_1.sel(time=np.isin(data_1[coord].values, data_2[coord].values))
    data_2 = data_2.sel(time=np.isin(data_2[coord].values, data_1[coord].values))
    return data_1, data_2


def standardize_spatial_orientation(
    ds: xr.Dataset,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.Dataset:
    """
    Standardize spatial orientation:

    - latitude: increasing (South -> North)
    - longitude: increasing (West -> East)
    """

    if lat_dim in ds.dims or lat_dim in ds.coords:
        if ds[lat_dim].size > 1 and ds[lat_dim][0] > ds[lat_dim][-1]:
            ds = ds.sortby(lat_dim)

    if lon_dim in ds.dims or lon_dim in ds.coords:
        if ds[lon_dim].size > 1 and ds[lon_dim][0] > ds[lon_dim][-1]:
            ds = ds.sortby(lon_dim)

    return ds

def load_tensor_data_from_netcdf_filepaths(
    path_predictors: str,
    path_predictand: str,
    predictand_var_name: str,
    period: Optional[tuple] = None,
    add_seasonal_predictors: bool = True,
    input_padding: Optional[tuple] = None,
    return_target_dates: bool = False,
    dates: np.ndarray = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads the data as torch.Tensor privided the filepath
    to NetCDF files.
    
    The data in the NetCDF files is expected to be raw.
    """
    ds_predictors = xr.open_dataset(path_predictors)
    ds_predictors = ds_predictors.sel(
        time=~((ds_predictors.time.dt.month == 2) & (ds_predictors.time.dt.day == 29))
    )
    
    ds_predictand = xr.open_dataset(path_predictand)
    ds_predictand = ds_predictand.sel(
        time=~((ds_predictand.time.dt.month == 2) & (ds_predictand.time.dt.day == 29))
    )
    if "Time" in ds_predictand.coords:
        ds_predictand = ds_predictand.rename({"Time": "time"}).sortby("time")
    ds_predictand = ds_predictand.assign_coords(time=ds_predictand.time.dt.floor("D"))
    
    if period is not None:
        ds_predictors = ds_predictors.sel(time=slice(*period))
        ds_predictand = ds_predictand.sel(time=slice(*period))

    if dates:
        ds_predictors = ds_predictors.sel(time=dates)
        ds_predictand = ds_predictand.sel(time=dates)
    
    ds_predictors = standardize_spatial_orientation(ds_predictors)
    ds_predictand = standardize_spatial_orientation(ds_predictand)
    ds_predictand, ds_predictors = align_datasets(ds_predictand, ds_predictors, "time")

    ds_x1d = get_x1d(ds_predictors, add_seasonal_predictors)

    x2d_varnames = [i for i in ds_predictors.data_vars]
    x2d_np = np.stack([ds_predictors[v].values for v in x2d_varnames], axis=1)  # (T, V, H, W)
    x1d_varnames = [i for i in ds_x1d.data_vars]
    x1d_np = np.stack([ds_x1d[v].values for v in x1d_varnames], axis=1)  # (T, V*2)
    y_np = ds_predictand[predictand_var_name].values

    x2d = torch.tensor(x2d_np, dtype=torch.float32)
    x1d = torch.tensor(x1d_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)  # (T, H, W)

    if input_padding is not None:
        x2d = F.pad(x2d, input_padding, mode="replicate")

    if not return_target_dates:
        return x2d, x1d, y
    else:
        target_dates = ds_predictand.time.values
        return x2d, x1d, y,  target_dates


def load_tensor_data_from_datasets(
    ds_x2d: xr.Dataset,
    ds_predictand: xr.Dataset,
    predictand_var_name: str,
    period: Optional[tuple] = None,
    filter_feb29: bool = True,
    add_seasonal_predictors: bool = True,
    input_padding: Optional[tuple] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads the data as torch.Tensor privided the xarray 
    datasets.
    
    The data in the datasets is expected to be raw.
    """
    if "Time" in ds_predictand.coords:
        ds_predictand = ds_predictand.rename({"Time": "time"}).sortby("time")
    ds_predictand = ds_predictand.assign_coords(time=ds_predictand.time.dt.floor("D"))
    
    if filter_feb29:
        ds_predictand = ds_predictand.sel(
            time=~((ds_predictand.time.dt.month == 2) & (ds_predictand.time.dt.day == 29))
        )
        ds_x2d = ds_x2d.sel(
            time=~((ds_x2d.time.dt.month == 2) & (ds_x2d.time.dt.day == 29))
        )

    if period is not None:
        ds_x2d = ds_x2d.sel(time=slice(*period))
        ds_predictand = ds_predictand.sel(time=slice(*period))

    ds_predictors = standardize_spatial_orientation(ds_predictors)
    ds_predictand = standardize_spatial_orientation(ds_predictand)
    ds_predictand, ds_x2d = align_datasets(ds_predictand, ds_x2d, "time")
    
    ds_x1d = get_x1d(ds_x2d, add_seasonal_predictors)

    x2d_varnames = [i for i in ds_x2d.data_vars]
    x2d_np = np.stack([ds_x2d[v].values for v in x2d_varnames], axis=1)  # (T, V, H, W)
    x1d_varnames = [i for i in ds_x1d.data_vars]
    x1d_np = np.stack([ds_x1d[v].values for v in x1d_varnames], axis=1)  # (T, V*2)
    predictand_np = ds_predictand[predictand_var_name].values

    x2d = torch.tensor(x2d_np, dtype=torch.float32)
    x1d = torch.tensor(x1d_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)  # (T, H, W)

    if input_padding is not None:
        x2d = F.pad(x2d, input_padding, mode="replicate")
    
    return x2d, x1d, y

