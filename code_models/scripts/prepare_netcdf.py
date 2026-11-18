
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

def load_coords_from_netcdf_filepaths(
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

    lons_target = ds_predictand.lon.values
    lats_target = ds_predictand.lat.values
    time = ds_predictand.time.values
    return time, lats_target, lons_target 


def main(SIM):

    ROOT_SIM_FOLDER = Path("../final_models")
    ROOT_MODEL_FOLDER = Path("../trained_model_parameters")

    SIM_FOLDER = ROOT_SIM_FOLDER / SIM / "simulations"
    MODEL_FOLDER = ROOT_MODEL_FOLDER / SIM

    assert SIM_FOLDER.exists()
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

    dates, lats, lons = load_coords_from_netcdf_filepaths(
        path_predictors=predictors_fname, path_predictand=predictand_fname,
        predictand_var_name=predictand_var_name, period=test_period
    )

    simulation_files_list =  list(SIM_FOLDER.glob("./simulation_*.npy"))

    if len(simulation_files_list) == 0:
        raise Exception(f"No simulation file was found in {SIM_FOLDER.resolve()}")

    print(f"{len(simulation_files_list)} simulations found.")

    simulations_list = [np.load(f).squeeze().astype(np.float32)[:, np.newaxis, :, :] 
                            for f in simulation_files_list]
    simulation_np = np.concatenate(simulations_list, axis=1)

    del simulations_list
    gc.collect()
    print(simulation_np.shape)
    ds_sims = xr.Dataset({
        predictand_var_name: xr.DataArray(
            data = simulation_np,
            dims = ["time", "member", "lat", "lon"],
            coords = {
                "time": dates, 
                "member": np.arange(len(simulation_files_list)),
                "lat": lats,
                "lon": lons,
                }
        )
    })

    del simulation_np
    gc.collect()

    fname_out = SIM_FOLDER.parent / f"pred_test_{SIM}.nc"
    ds_sims.to_netcdf(fname_out)

    print("NetCDF saved in", fname_out.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--SIM", type=str, default=SIM_DEFAULT, help="Simulation name")
    args = parser.parse_args()
    main(args.SIM)
