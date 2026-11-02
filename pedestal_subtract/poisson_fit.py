"""
poisson_fit — dark current from a Gaussian (x) Poisson fit to the pixel-charge histogram.

Instead of reading the dark current off the one-electron peak amplitude, the whole
zero/one-electron histogram is modelled as the read-noise Gaussian convolved with a
Poisson(lambda) number of electrons per pixel, so lambda *is* the dark current
(electrons / image pixel / image). Pedestal subtraction and the histogram window reuse
the existing pipeline; only the fit is new.
"""
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import minimize
from scipy.special import gammaln, ndtr

from analysis_tools import stitch_fits

from .cli_config import (
    _coerce_fit_cols,
    _config_default,
    _derive_data_path,
    _load_config,
    _normalize_fit_cols,
    _overscan_ext_indices,
    _resolve_fit_col_masks,
    _wildcard_free_base,
)
from .constants import _MAX_GAIN_ADU, _MIN_GAIN_ADU, _ZERO_PEAK_LEFT_FIT_NSIGMA
from .fit_zero_one import get_zero_one_peaks_ext
from .fits_io import (
    get_exposure_time_days,
    get_fits,
    get_fits_header,
    get_pixel_binning,
    is_serial_register_image,
    overscan_cols_from_header,
)
from .pedestal import pedestal_subtract_ext_cached

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'pedestal_subtract_LBC.json'
_STITCHED_DIR = 'stitched-fits'


def poisson_gauss_model(edges, lam, s, m0, g, n_pix, kmax=3):
    """Expected counts per histogram bin for Poisson(lam) electrons + Gaussian noise ``s``.

    Peak k sits at ``m0 + k*g`` with weight Pois(k; lam); each bin integrates the
    Gaussians exactly (CDF differences), so it stays correct for wide bins.
    """
    ks = np.arange(kmax + 1)
    weights = np.exp(ks * np.log(lam) - lam - gammaln(ks + 1))
    cdf = ndtr((edges[:, None] - (m0 + ks * g)[None, :]) / s)
    return n_pix * (np.diff(cdf, axis=0) @ weights)


def _nll(params, centers_edges, counts, n_pix, kmax):
    """Binned Poisson negative log-likelihood; params = (log10 lam, s, m0, g)."""
    expected = poisson_gauss_model(centers_edges, 10 ** params[0], *params[1:], n_pix, kmax)
    return np.sum(expected - counts * np.log(np.maximum(expected, 1e-12)))


def _hessian(f, x, rel_step=1e-3):
    """Central-difference Hessian of ``f`` at ``x`` (steps scaled to each |x_i|)."""
    h = rel_step * np.maximum(np.abs(x), 1e-3)
    H = np.empty((x.size, x.size))
    for i in range(x.size):
        for j in range(i, x.size):
            di, dj = np.eye(x.size)[i] * h[i], np.eye(x.size)[j] * h[j]
            H[i, j] = H[j, i] = (f(x + di + dj) - f(x + di - dj)
                                 - f(x - di + dj) + f(x - di - dj)) / (4 * h[i] * h[j])
    return H


def fit_poisson_gauss(data, counts, edges, pedestal, noise, gain, popt, kmax=3, fixed=None):
    """Fit lambda, sigma, m0, gain to one extension's histogram (``counts``, ``edges``).

    ``pedestal, noise, gain, popt`` are the existing double-Gaussian fit's results
    (``get_zero_one_peaks_ext``), used only to seed the fit and pick the fit window; the
    same left-tail cut as that fit is applied. Returns a dict (NaNs if the seed fit found
    no trustworthy one-electron peak). ``lam`` is in e-/image-pixel/image. With
    ``fixed=(sigma_adu, m0, gain)`` those three are held fixed and only lambda is fit
    (for low-statistics subsets such as single image rows).
    """
    nan = dict(lam=np.nan, lam_err=np.nan, sigma_adu=np.nan, m0=np.nan, gain=np.nan,
               converged=False)
    if not np.isfinite(gain):
        return nan
    n_pix = int(np.isfinite(data).sum())
    keep = 0.5 * (edges[:-1] + edges[1:]) >= pedestal - _ZERO_PEAK_LEFT_FIT_NSIGMA * noise
    first = int(np.argmax(keep))
    e, c = edges[first:], counts[first:]

    lam0 = max(popt[4] / (popt[3] + popt[4]), 1e-8)
    x0 = np.array([np.log10(lam0), *(fixed if fixed is not None else (noise, pedestal, gain))])
    n_free = 4 if fixed is None else 1
    bounds = [(-8, 1), (0.2 * noise, 5 * noise), (pedestal - noise, pedestal + noise),
              (_MIN_GAIN_ADU, _MAX_GAIN_ADU)]
    f = lambda p: _nll(np.concatenate([p, x0[p.size:]]), e, c, n_pix, kmax)
    res = minimize(f, x0[:n_free], method='L-BFGS-B', bounds=bounds[:n_free])
    res.x = np.concatenate([res.x, x0[n_free:]])

    try:
        cov = np.linalg.inv(_hessian(f, res.x[:n_free]))
        lam_err = 10 ** res.x[0] * np.log(10) * np.sqrt(cov[0, 0])
    except (np.linalg.LinAlgError, ValueError):
        lam_err = np.nan
    return dict(lam=10 ** res.x[0], lam_err=lam_err, sigma_adu=res.x[1], m0=res.x[2],
                gain=res.x[3], converged=bool(res.success))


def _stitched_path(image_path, verbose):
    """Resolve ``image_path`` (glob) to one FITS file, stitching multiple files like the CLI."""
    input_dir, pattern = _derive_data_path(image_path)
    if _STITCHED_DIR in input_dir.parts:
        match = next(input_dir.glob(pattern), None)
        if match is None:
            raise FileNotFoundError(f'no stitched FITS matches: {image_path}')
        return match
    out_path = (_wildcard_free_base(image_path) / _STITCHED_DIR).resolve()
    stitched = stitch_fits(input_dir / pattern, out_path=out_path, print_header=verbose)
    if stitched is None:
        raise FileNotFoundError(f'could not stitch files matching: {image_path}')
    return Path(stitched)


def analyze_image(image_path, config=_DEFAULT_CONFIG, verbose=False, by_row=False, **overrides):
    """Pedestal-subtract ``image_path`` and fit Poisson (x) Gaussian per extension.

    ``image_path`` may be an already-stitched FITS or a glob of per-image files (which
    are stitched first). ``config`` is a pedestal_subtract JSON path or dict; its
    pedestal-subtraction / histogram keys are used, and any key can be overridden by
    keyword. Returns one dict per extension: ``ext, lam, lam_err, sigma_adu, noise_e,
    m0, gain, converged, n_pix, exposure_days, pixel_binning, counts, edges``.

    With ``by_row=True`` each dict also gets ``exposure_s`` (shutter, header ``EXPOSURE``),
    ``n_rows`` (rows per stitched image) and ``row_fits``: for each row index, the pixels
    from that row of every stitched image are pooled and only lambda is fit (sigma, m0 and
    gain fixed at the whole-image fit), with the effective exposure
    ``t_eff_s = EXPOSURE + row * (T_total - EXPOSURE) / n_rows`` (T_total = DATEEND - DATEINI).
    """
    cfg = _load_config(config) if not isinstance(config, dict) else dict(config)
    cfg.update(overrides)
    get = lambda key, fallback: _config_default(cfg, key, fallback)

    file_path = _stitched_path(image_path, verbose)
    raw = get_fits(str(file_path))
    n_ext = len(raw)
    n_std = get('n_std_to_mask', 0.5)
    fit_cols_ext = _resolve_fit_col_masks(
        _normalize_fit_cols(_coerce_fit_cols(get('fit_cols', None)), n_ext), raw, n_std)

    overscan_range = overscan_cols_from_header(get_fits_header(file_path)) or tuple(
        get('overscan_cols', (-50, None)))
    overscan_exts = _overscan_ext_indices(get('use_overscan_only', False), n_ext)
    data_ext = pedestal_subtract_ext_cached(
        raw, source_path=file_path, n_std_to_mask=n_std,
        axis=get('pedestal_subtraction_axis', 'row_then_col'),
        use_biweight_loc=get('use_biweight_loc', True),
        use_biweight_midvar=get('use_biweight_midvar', True),
        max_iter=get('pedsub_max_iter', 5), cache_dir=get('pedsub_cache_dir', None),
        force=get('force_pedsub', False), verbose=verbose,
        overscan_cols=[overscan_range if i in overscan_exts else None for i in range(n_ext)])
    data_ext = [np.asarray(d)[:, m] if m is not None else d for d, m in zip(data_ext, fit_cols_ext)]

    counts_ext, edges_ext, pedestals, gains, popts, _ = get_zero_one_peaks_ext(
        data_ext, n_bins=get('zero_one_n_bins', 75),
        window_left_scale=get('zero_one_window_left_scale', 1.0),
        window_right_scale=get('zero_one_window_right_scale', 1.0),
        gain_seed=get('zero_one_gain_guess', None))

    exposure_days = (get('exposure_time_s', None) / 86400 if get('exposure_time_s', None) is not None
                     else get_exposure_time_days(file_path))
    npbin, nsbin = get_pixel_binning(file_path)
    pixel_binning = (nsbin or 1) if is_serial_register_image(file_path) else (npbin or 1) * (nsbin or 1)

    rows = []
    for ext, (data, counts, edges, ped, gain, popt) in enumerate(
            zip(data_ext, counts_ext, edges_ext, pedestals, gains, popts)):
        fit = fit_poisson_gauss(data, counts, edges, ped, popt[0], gain, popt, get('poisson_kmax', 3))
        rows.append(dict(ext=ext + 1, **fit, noise_e=fit['sigma_adu'] / fit['gain'],
                         n_pix=int(np.isfinite(data).sum()), exposure_days=exposure_days,
                         pixel_binning=pixel_binning, counts=counts, edges=edges))
        if by_row and np.isfinite(fit['gain']):
            n_rows = data.shape[0] // fits.getheader(file_path, 1).get('STITCHED', 1)
            exp_s = float(get_fits_header(file_path)['EXPOSURE'])
            shape = (fit['sigma_adu'], fit['m0'], fit['gain'])
            row_fits = []
            for r in range(n_rows):
                d = np.asarray(data[r::n_rows])
                d = d[np.isfinite(d)]
                rf = fit_poisson_gauss(d, np.histogram(d, bins=edges)[0], edges, ped, popt[0], gain, popt,
                                       get('poisson_kmax', 3), fixed=shape)
                row_fits.append(dict(row=r, t_eff_s=exp_s + r * (exposure_days * 86400 - exp_s) / n_rows,
                                     lam=rf['lam'], lam_err=rf['lam_err'], n_pix=d.size))
            rows[-1].update(row_fits=row_fits, exposure_s=exp_s, n_rows=n_rows)
        if verbose:
            print(f"ext {ext + 1}: lambda = {fit['lam']:.3e} +/- {fit['lam_err']:.1e} e-/pix, "
                  f"sigma = {rows[-1]['noise_e']:.3f} e-, gain = {fit['gain']:.3f} ADU/e-")
    return rows
