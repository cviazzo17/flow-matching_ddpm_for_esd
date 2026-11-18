# src/statistical_downscaling/evaluation/metrics/univariate/deterministic.py

import numpy as np
import xarray as xr
from typing import Union, Optional, Sequence


_AD_SIG = np.array([0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001])
_AD_CRIT_K2 = np.array([0.325, 1.226, 1.961, 2.718, 3.752, 4.592, 6.546])
_AD_PFIT = np.polyfit(_AD_CRIT_K2, np.log(_AD_SIG), 2)

def mae(preds: Union[xr.Dataset, xr.DataArray], target: Union[xr.Dataset, xr.DataArray]) -> xr.Dataset:
    preds = preds.copy()
    if "member" in preds.dims:
        preds = preds.mean("member")
    mae = (np.abs(preds - target)).mean(dim="time")
    return mae
    
    
def rmse(preds: Union[xr.Dataset, xr.DataArray], target: Union[xr.Dataset, xr.DataArray]) -> xr.Dataset:
    preds = preds.copy()
    if "member" in preds.dims:
        preds = preds.mean("member")
        
    rmse = np.sqrt( ( (preds - target) ** 2 ).mean(dim="time"))
    return rmse

# ---------------------- 

def ecdf(x: np.ndarray, pseudo: bool = False, 
         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Computes the ECDF as ranks / (T + 1). If pseudo = True ties
    are broken at random (e.g. dry days)."""
    
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
    

def wasserstein_1d(preds: xr.DataArray, target: xr.DataArray, dim: str = "time") -> xr.DataArray:
    def _get_wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
        """W1 = ∫ |F(z) - G(z)| dz = area between the two ECDFs."""
        x = np.sort(np.asarray(x, dtype=float))
        y = np.sort(np.asarray(y, dtype=float))
        z = np.unique(np.concatenate([x, y]))
        F = np.searchsorted(x, z, side="right") / x.size
        G = np.searchsorted(y, z, side="right") / y.size
        # Rectangles: height |F - G| on [z_k, z_{k+1}), width z_{k+1} - z_k
        return float(np.sum(np.abs(F - G)[:-1] * np.diff(z)))

    def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size == 0 or y.size == 0:
            return np.nan
        return _get_wasserstein_distance(x, y)
    
    return xr.apply_ufunc(
        _wasserstein_1d,
        preds,
        target,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        exclude_dims={dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    ).rename("wasserstein")


def ks_test_1d(
    preds: Union[xr.Dataset, xr.DataArray],
    target: Union[xr.Dataset, xr.DataArray],
    dim: str = "time",
) -> xr.Dataset:
    """Two-sample KS test along `dim` at every grid point.
    DataArray -> Dataset with variables `ks` and `pval` (dims lat, lon).
    Dataset   -> one pair per common variable: `<var>_ks` and `<var>_pval`.
    """
    
    def _get_ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
        """D = max |F(z) - G(z)| over the distinct values of the pooled sample."""
        x = np.sort(np.asarray(x, dtype=float))
        y = np.sort(np.asarray(y, dtype=float))
        n, m = x.size, y.size
        z = np.unique(np.concatenate([x, y]))
        F = np.searchsorted(x, z, side="right") / n
        G = np.searchsorted(y, z, side="right") / m
        return float(np.max(np.abs(F - G)))


    def _ks_asymptotic_pvalue(statistic: float, n: int, m: int, terms: int = 100) -> float:
        """Kolmogorov distribution: P(D > d) ≈ 2 Σ (-1)^(k-1) exp(-2 k² λ²),
        with λ = (en + 0.12 + 0.11/en)·D, en = sqrt(nm/(n+m)) (Stephens' correction)."""
        en = np.sqrt(n * m / (n + m))
        lam = (en + 0.12 + 0.11 / en) * statistic
        k = np.arange(1, terms + 1)
        p = 2 * np.sum((-1) ** (k - 1) * np.exp(-2 * k**2 * lam**2))
        return float(np.clip(p, 0.0, 1.0))


    def _ks_1d(x: np.ndarray, y: np.ndarray):
        """KS test for ONE grid point: x and y are 1-D time series.
        NaNs are removed first (ocean, missing data...), so n and m can
        differ from point to point."""
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        n, m = x.size, y.size
        if n == 0 or m == 0:
            return np.nan, np.nan
        D = _get_ks_statistic(x, y)
        return D, _ks_asymptotic_pvalue(D, n, m)


    def _ks_dataarray(preds: xr.DataArray, target: xr.DataArray, dim: str) -> xr.Dataset:
        ks, pval = xr.apply_ufunc(
            _ks_1d,
            preds,
            target,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[], []],
            exclude_dims={dim},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32, np.float32],
        )
        return xr.Dataset({"ks": ks.astype(np.float32), "pval": pval.astype(np.float32)})
    
    if isinstance(preds, xr.DataArray) and isinstance(target, xr.DataArray):
        return _ks_dataarray(preds, target, dim)

    if isinstance(preds, xr.Dataset) and isinstance(target, xr.Dataset):
        out = {}
        for var in preds.data_vars:
            if var not in target.data_vars:
                continue
            res = _ks_dataarray(preds[var], target[var], dim)
            out[f"{var}_ks"] = res["ks"]
            out[f"{var}_pval"] = res["pval"]
        return xr.Dataset(out)

    raise TypeError("preds and target must both be DataArray or both be Dataset")


def ad_test_1d(
    preds: Union[xr.Dataset, xr.DataArray],
    target: Union[xr.Dataset, xr.DataArray],
    dim: str = "time",
) -> xr.Dataset:
    """Two-sample Anderson-Darling test along `dim` at every grid point.
    `ad` is the standardized statistic T (same as scipy.stats.anderson_ksamp).
    DataArray -> Dataset with variables `ad` and `pval` (dims lat, lon).
    Dataset   -> one pair per common variable: `<var>_ad` and `<var>_pval`.
    """
    
    def _get_ad_statistic(x: np.ndarray, y: np.ndarray) -> float:
        """A² of Scholz & Stephens with the midrank correction for ties.
        The sum runs over the DISTINCT values z*_1 < ... < z*_L of the pooled sample:
            A² = (N-1)/N² Σ_i (1/n_i) Σ_j l_j (N·Ma_ij - n_i·Ba_j)² / (Ba_j·(N-Ba_j) - N·l_j/4)
        """
        x = np.sort(np.asarray(x, dtype=float))
        y = np.sort(np.asarray(y, dtype=float))
        N = x.size + y.size

        z, l = np.unique(np.concatenate([x, y]), return_counts=True)
        Ba = np.cumsum(l) - l / 2.0
        den = Ba * (N - Ba) - N * l / 4.0

        A2 = 0.0
        for s in (x, y):
            ni = s.size
            le = np.searchsorted(s, z, side="right")   # obs. of this sample <= z
            lt = np.searchsorted(s, z, side="left")    # obs. of this sample <  z
            Ma = (le + lt) / 2.0                       # midrank count (= le - f/2)
            A2 += np.sum(l * (N * Ma - ni * Ba) ** 2 / den) / ni
        return float(A2 * (N - 1) / N**2)
    
    def _ad_standardize(A2: float, n: int, m: int) -> float:
        """T = (A² - (k-1)) / σ_N, with the exact variance of Scholz & Stephens (k = 2)."""
        k = 2
        N = n + m
        H = 1.0 / n + 1.0 / m
        inv = 1.0 / np.arange(1, N)                    # 1/1 ... 1/(N-1)
        h = inv.sum()
        # g = Σ_{i=1}^{N-2} Σ_{j=i+1}^{N-1} 1/((N-i)·j) = Σ_i (h - cumsum_i) / (N - i)
        i = np.arange(1, N - 1)
        g = np.sum((h - np.cumsum(inv)[: N - 2]) / (N - i))

        a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * H
        b = (2 * g - 4) * k**2 + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
        c = (6 * h + 2 * g - 2) * k**2 + (4 * h - 4 * g + 6) * k + (2 * h - 6) * H + 4 * h
        d = (2 * h + 6) * k**2 - 4 * h * k
        var = (a * N**3 + b * N**2 + c * N + d) / ((N - 1) * (N - 2) * (N - 3))
        return float((A2 - (k - 1)) / np.sqrt(var))


    def _ad_pvalue(T: float) -> float:
        """Interpolate p from the critical-value table (same method as scipy).
        Only valid in [0.001, 0.25]: values outside are capped."""
        p = np.exp(np.polyval(_AD_PFIT, T))
        return float(np.clip(p, 0.001, 0.25))


    def _ad_1d(x: np.ndarray, y: np.ndarray):
        """AD test for ONE grid point: x and y are 1-D time series (NaNs removed)."""
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        n, m = x.size, y.size
        if n < 2 or m < 2 or np.unique(np.concatenate([x, y])).size < 2:
            return np.nan, np.nan
        T = _ad_standardize(_get_ad_statistic(x, y), n, m)
        return T, _ad_pvalue(T)


    def _ad_dataarray(preds: xr.DataArray, target: xr.DataArray, dim: str) -> xr.Dataset:
        ad, pval = xr.apply_ufunc(
            _ad_1d,
            preds,
            target,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[], []],
            exclude_dims={dim},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32, np.float32],
        )
        return xr.Dataset({"ad": ad.astype(np.float32), "pval": pval.astype(np.float32)})
    if isinstance(preds, xr.DataArray) and isinstance(target, xr.DataArray):
        return _ad_dataarray(preds, target, dim)

    if isinstance(preds, xr.Dataset) and isinstance(target, xr.Dataset):
        out = {}
        for var in preds.data_vars:
            if var not in target.data_vars:
                continue
            res = _ad_dataarray(preds[var], target[var], dim)
            out[f"{var}_ad"] = res["ad"]
            out[f"{var}_pval"] = res["pval"]
        return xr.Dataset(out)

    raise TypeError("preds and target must both be DataArray or both be Dataset")
    
    
    
def fdr_reject(
    pval: xr.DataArray,
    alpha_fdr: float = 0.10,
    spatial_dims: Sequence[str] = ("lat", "lon"),
) -> xr.DataArray:
    """Field significance with the False Discovery Rate (Benjamini & Hochberg 1995,
    as recommended by Wilks 2016 for climate fields).

    The N valid p-values of the field are sorted, p_(1) <= ... <= p_(N), and
        p_FDR = max{ p_(i) : p_(i) <= (i / N) * alpha_fdr }.
    A grid point rejects H0 if p <= p_FDR. If no p_(i) satisfies the condition,
    no point rejects.

    alpha_fdr = 2 * alpha (e.g. 0.10 for alpha = 0.05) is Wilks' (2016) recommendation
    for spatially correlated fields.

    Returns 1 (reject), 0 (do not reject) or NaN (no data), with the same dims as
    `pval`. Any extra dim (e.g. member) is treated independently.
    The threshold p_FDR is stored in the attrs when there are no extra dims.
    """
    spatial_dims = list(spatial_dims)

    def _bh(p):
        shape = p.shape
        p = p.ravel().astype(float)
        out = np.full(p.shape, np.nan)
        valid = np.isfinite(p)
        pv = p[valid]
        N = pv.size
        if N == 0:
            return out.reshape(shape), np.nan
        ps = np.sort(pv)
        ok = ps <= alpha_fdr * np.arange(1, N + 1) / N
        p_fdr = ps[ok].max() if ok.any() else 0.0
        out[valid] = (pv <= p_fdr) if ok.any() else 0.0
        return out.reshape(shape), p_fdr

    reject, p_fdr = xr.apply_ufunc(
        _bh,
        pval,
        input_core_dims=[spatial_dims],
        output_core_dims=[spatial_dims, []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32, np.float32],
    )
    reject = reject.rename("reject_fdr")
    if p_fdr.ndim == 0:
        reject.attrs["p_fdr"] = float(p_fdr)
    reject.attrs["alpha_fdr"] = alpha_fdr
    return reject

