# src/statistical_downscaling/evaluation/stats/univariate/ralsd.py
"""
Radially Averaged Log Spectral Distance (RALSD) between a ground truth and a
prediction (Harris et al., 2022).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

EARTH_RADIUS_KM = 6371.0

def grid_spacing_km(da: xr.DataArray, y: str = "lat", x: str = "lon") -> tuple[float, float]:
    """(dy, dx) in km for a regular lat/lon grid; dx uses cos(mean latitude)."""
    lat, lon = da[y].values, da[x].values
    dlat, dlon = np.diff(lat), np.diff(lon)
    if not (np.allclose(dlat, dlat[0], rtol=1e-3) and np.allclose(dlon, dlon[0], rtol=1e-3)):
        raise ValueError("Grid is not regular; pass dx and dy explicitly.")
    dy = EARTH_RADIUS_KM * np.deg2rad(abs(dlat[0]))
    dx = EARTH_RADIUS_KM * np.deg2rad(abs(dlon[0])) * np.cos(np.deg2rad(lat.mean()))
    return float(dy), float(dx)


def _window2d(ny, nx, kind):
    if kind == "none":
        return np.ones((ny, nx))
    from scipy.signal import windows
    if kind == "hann":
        return np.outer(windows.hann(ny, sym=False), windows.hann(nx, sym=False))
    if kind == "tukey":
        return np.outer(windows.tukey(ny, 0.25), windows.tukey(nx, 0.25))
    raise ValueError("window must be 'hann', 'tukey' or 'none'.")


def _radial_bins(ny, nx, dy, dx):
    """Radial bin of every 2D wavenumber (0 = discarded), bin centres, bin width."""
    ky = np.fft.fftfreq(ny, d=dy)
    kx = np.fft.fftfreq(nx, d=dx)
    kk = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    dk = max(1.0 / (ny * dy), 1.0 / (nx * dx))
    nbins = int(np.floor(min(0.5 / dy, 0.5 / dx) / dk))
    idx = np.floor(kk / dk + 0.5).astype(int)
    idx[(idx < 1) | (idx > nbins)] = 0  
    return idx.ravel(), np.arange(1, nbins + 1) * dk, dk, nbins



def _psd_np(f, dy, dx, window, nan, chunk=128):
    """f: (..., ny, nx) -> E (..., K). Processes `chunk` fields at a time (low RAM)."""
    lead, (ny, nx) = f.shape[:-2], f.shape[-2:]
    f = f.reshape(-1, ny, nx)
    n = f.shape[0]

    w = _window2d(ny, nx, window).astype(np.float32)
    wnorm = float((w.astype(float) ** 2).mean())
    idx, _, dk, nbins = _radial_bins(ny, nx, dy, dx)

    # rfft2 keeps only kx >= 0: columns with a conjugate twin count twice
    idx_r = idx.reshape(ny, nx)[:, : nx // 2 + 1].ravel()
    wt = np.full(nx // 2 + 1, 2.0)
    wt[0] = 1.0
    if nx % 2 == 0:
        wt[-1] = 1.0
    wt = np.broadcast_to(wt, (ny, nx // 2 + 1)).ravel()
    keep = idx_r > 0
    idx_k, wt_k = idx_r[keep] - 1, wt[keep]
    cnt = np.bincount(idx_k, weights=wt_k, minlength=nbins)        # modes per ring (uneven!)
    centres = np.arange(1, nbins + 1) * dk
    ring = 2 * np.pi * centres * (nx * dx) * (ny * dy)             # expected modes / dk = 2πk/(δkx δky)
    scale = ring / (np.maximum(cnt, 1) * (ny * nx) ** 2 * wnorm)

    E = np.full((n, nbins), np.nan)
    for s in range(0, n, chunk):
        g = np.asarray(f[s:s + chunk], dtype=np.float32)
        bad = ~np.isfinite(g)
        all_bad, any_bad = bad.all(axis=(1, 2)), bad.any(axis=(1, 2))
        if nan == "mean":
            m = np.nanmean(np.where(all_bad[:, None, None], 0.0, g), axis=(1, 2), keepdims=True)
            g = np.where(bad, m, g)
        else:
            g = np.where(bad, 0.0, g)
        g -= g.mean(axis=(1, 2), keepdims=True)
        g *= w
        P = np.abs(np.fft.rfft2(g)) ** 2                           # (c, ny, nx//2+1)
        P = P.reshape(P.shape[0], -1)[:, keep] * wt_k
        Ec = np.zeros((P.shape[0], nbins))
        for b in range(nbins):                                     # nbins ~ 50: cheap
            Ec[:, b] = P[:, idx_k == b].sum(axis=1)
        Ec *= scale
        Ec[all_bad] = np.nan
        if nan == "drop":
            Ec[any_bad] = np.nan
        E[s:s + chunk] = Ec
    return E.reshape(lead + (nbins,))


def isotropic_psd(
    da: xr.DataArray,
    y: str = "lat",
    x: str = "lon",
    dy: float | None = None,
    dx: float | None = None,
    window: str = "hann",
    nan: str = "zero",
) -> xr.DataArray:
    """
    Isotropic PSD of every 2D field (y, x); all other dims are kept.

    window : 'hann' (default), 'tukey' or 'none'. The domain is not periodic:
             without a window the edge jump leaks power into small scales
             (non-periodic k^-3 test field: hann -3.09, tukey -3.61, none -1.86).
    nan    : 'zero' (default; NaN -> 0, right for masked dry days),
             'mean' (NaN -> field mean) or 'drop' (fields with any NaN -> NaN).

    Works lazily with dask if `da` is chunked along non-spatial dims
    (lat/lon must be a single chunk).
    """
    if dy is None or dx is None:
        dy, dx = grid_spacing_km(da, y, x)
    ny, nx = da.sizes[y], da.sizes[x]
    _, centres, _, nbins = _radial_bins(ny, nx, dy, dx)

    E = xr.apply_ufunc(
        _psd_np, da,
        input_core_dims=[[y, x]], output_core_dims=[["k"]],
        kwargs=dict(dy=dy, dx=dx, window=window, nan=nan),
        dask="parallelized", output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"k": nbins}},
    )
    E = E.assign_coords(k=centres, wavelength=("k", 1.0 / centres)).rename("psd")
    E["k"].attrs["units"] = "cycles/km"
    E["wavelength"].attrs["units"] = "km"
    E.attrs.update(window=window, nan=nan, dy_km=dy, dx_km=dx)
    return E


# ----------------------------------------------------------------------------
# season filter, mean spectra and RALSD
# ----------------------------------------------------------------------------
def _filter_by_season(ds, season):
    return ds if season is None else ds.sel(time=ds["time.season"] == season)

def psd(x0, x1, var, season=None, convention="rapsd", window="hann",
        chunk_time=365, member_dim="member"):
    """
    Mean spatial power spectrum of ground truth (x0) and prediction (x1).

    convention : 'rapsd' (mean power per ring, like pySTEPS / Harris et al.)
                 or 'isotropic' (E(k) = 2*pi*k * rapsd; sum E dk ~ variance).
    Members (if any) are averaged together with time.
    Returns two DataArrays with dim 'wavenumber' (cycles/km) and coord
    'wavelength' (km).
    """
    out = []
    for x, name in ((x0, "PSD_x0"), (x1, "PSD_x1")):
        da = _filter_by_season(x, season)[var]
        if "time" in da.dims and chunk_time:
            da = da.chunk({"time": chunk_time})
        E = isotropic_psd(da, window=window, nan="zero")
        dims = [d for d in ("time", member_dim) if d in E.dims]
        E = E.mean(dims, skipna=True).compute()
        if convention == "rapsd":
            E = E / (2 * np.pi * E["k"])
        elif convention != "isotropic":
            raise ValueError("convention must be 'rapsd' or 'isotropic'.")
        out.append(E.rename({"k": "wavenumber"}).rename(name))
    return tuple(out)


def ralsd(psd_x0, psd_x1, min_wavelength=None, root=True):
    """
    Radially Averaged Log Spectral Distance (Harris et al., 2022), in dB:
        sqrt( mean_k [10 log10(P_x0 / P_x1)]^2 )      (root=True)
    min_wavelength : ignore scales shorter than this (km), e.g. 4*dx.
    """
    p0, p1 = xr.align(psd_x0, psd_x1, join="inner")
    if min_wavelength is not None:
        keep = p0["wavelength"] >= min_wavelength
        p0, p1 = p0.where(keep, drop=True), p1.where(keep, drop=True)
    lr = 10.0 * np.log10(np.maximum(p0, 1e-30) / np.maximum(p1, 1e-30))
    msd = float((lr ** 2).mean())
    return float(np.sqrt(msd)) if root else msd



def psd_mean(E: xr.DataArray, dims=("time", "member")) -> xr.DataArray:
    """Mean spectrum over `dims` (only those present). NaN fields are skipped."""
    d = [x for x in dims if x in E.dims]
    return E.mean(d, skipna=True) if d else E
 
 
def psd_ratio(E_model: xr.DataArray, E_ref: xr.DataArray,
              dims=("time", "member")) -> xr.DataArray:
    """Ratio of mean spectra: < 1 at short wavelengths = model too smooth."""
    return (psd_mean(E_model, dims) / psd_mean(E_ref, dims)).rename("psd_ratio")


