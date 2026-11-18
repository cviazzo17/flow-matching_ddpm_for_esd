import sys
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Optional
import importlib
import argparse
import gc


sys.path.append("..")
SIM_DEFAULT = "FlowDEFprFlowMatching_sqrTransforms_s0_b68803X1bf0b7"

N_DAYS = 100 # number of days to select

# utilities ---------------------------------------------------------------------

def align_datasets(
    data_1: xr.Dataset, data_2: xr.Dataset, coord: str
) -> tuple[xr.Dataset, xr.Dataset]:
    data_1 = data_1.sel(time=np.isin(data_1[coord].values, data_2[coord].values))
    data_2 = data_2.sel(time=np.isin(data_2[coord].values, data_1[coord].values))
    return data_1, data_2


def standardize_spatial_orientation(
    ds: xr.Dataset,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.Dataset:
    if lat_dim in ds.dims or lat_dim in ds.coords:
        if ds[lat_dim].size > 1 and ds[lat_dim][0] > ds[lat_dim][-1]:
            ds = ds.sortby(lat_dim)

    if lon_dim in ds.dims or lon_dim in ds.coords:
        if ds[lon_dim].size > 1 and ds[lon_dim][0] > ds[lon_dim][-1]:
            ds = ds.sortby(lon_dim)

    return ds


def load_target_dataset_from_netcdf_filepaths(
    path_predictors: str,
    path_predictand: str,
    predictand_var_name: str,
    period: Optional[tuple] = None
) -> tuple:
    ds_predictors = xr.open_dataset(path_predictors)
    ds_predictors = ds_predictors.sel(
        time=~((ds_predictors.time.dt.month == 2) & (ds_predictors.time.dt.day == 29))
    )

    ds_predictand = xr.open_dataset(path_predictand)
    ds_predictand = ds_predictand.sel(
        time=~((ds_predictand.time.dt.month == 2) & (ds_predictand.time.dt.day == 29))
    )

    ds_predictand = ds_predictand.assign_coords(time=ds_predictand.time.dt.floor("D"))

    if period is not None:
        ds_predictors = ds_predictors.sel(time=slice(*period))
        ds_predictand = ds_predictand.sel(time=slice(*period))

    ds_predictors = standardize_spatial_orientation(ds_predictors)
    ds_predictand = standardize_spatial_orientation(ds_predictand)
    ds_predictand, ds_predictors = align_datasets(ds_predictand, ds_predictors, "time")

    return ds_predictand


# logic -------------------------------------------------------------------------------

def _get_spatially_accumulated_pr(da: xr.DataArray) -> xr.DataArray:
    return da.sum(dim=("lat", "lon"))


def get_n_more_precipitating_dates(
    da: xr.DataArray, n: int = 100, return_indeces: bool = True
) -> np.ndarray:
    spatially_accumulated_pr = _get_spatially_accumulated_pr(da)
    spatially_accumulated_pr = spatially_accumulated_pr.compute()

    n_more_precipitating_days = spatially_accumulated_pr.sortby(
        spatially_accumulated_pr, ascending=False
        ).isel(time=slice(0, n, 1))

    print("min:", np.round(n_more_precipitating_days.min().values, 2), flush=True)
    print("max:", np.round(n_more_precipitating_days.max().values, 2), flush=True)
    print("mean:", np.round(n_more_precipitating_days.mean().values, 2), flush=True)
    print("median:", np.round(n_more_precipitating_days.quantile(q=0.5).values,  2), flush=True)
    print("p10:", np.round(n_more_precipitating_days.quantile(q=0.1).values,  2), flush=True)
    print("p90:", np.round(n_more_precipitating_days.quantile(q=0.9).values, 2), flush=True)

    selected_dates = n_more_precipitating_days.sortby("time", ascending=True).time.values

    if not return_indeces:
        return selected_dates

    all_dates = da["time"].values
    indices = np.searchsorted(all_dates, selected_dates)

    assert np.array_equal(all_dates[indices], selected_dates), (
        "Index lookup mismatch: da['time'] is not sorted ascending as expected."
    )

    return selected_dates, indices


# main ---------------------------------------------------------------------------------


def main(SIM):

    ROOT_MODEL_FOLDER = Path("../trained_model_parameters")
    MODEL_FOLDER = ROOT_MODEL_FOLDER / SIM

    assert MODEL_FOLDER.exists()

    CONFIG_SIM_FILE = list((MODEL_FOLDER).glob("./CFG_*.py"))[0].resolve()

    print("Simulation:", SIM),
    print("Config file:", CONFIG_SIM_FILE)

    spec = importlib.util.spec_from_file_location("module_config", CONFIG_SIM_FILE)
    module_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_config)


    predictand_fname = Path("../data/predictand") / Path(module_config.predictand_fname).name
    predictors_fname = Path("../data/predictors") / Path(module_config.predictors_fname).name
    predictand_var_name = module_config.varname_dataset

    test_period = module_config.test_period

    ds_target_test = load_target_dataset_from_netcdf_filepaths(
        path_predictors=predictors_fname, path_predictand=predictand_fname,
        predictand_var_name=predictand_var_name, period=test_period,
    )

    n_more_precipitating_days, n_more_precipitating_indices = get_n_more_precipitating_dates(
        ds_target_test[predictand_var_name], n=N_DAYS
    )

    print("Test period size:", ds_target_test.time.size, flush=True)
    print("N selected days:", n_more_precipitating_days.size, flush=True)
    print("porcentage of selected days:", np.round(len(n_more_precipitating_days) / ds_target_test.time.size * 100, 2), flush=True)

    np.save(f"../data/{N_DAYS}_more_precipitating_days.npy", n_more_precipitating_days)
    np.save(f"../data/{N_DAYS}_more_precipitating_days_indices.npy", n_more_precipitating_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--SIM", type=str, default=SIM_DEFAULT, help="Simulation name")
    args = parser.parse_args()
    main(args.SIM)
