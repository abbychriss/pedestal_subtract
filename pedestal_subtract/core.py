"""
Core analysis routines for the pedestal-subtraction + double-Gaussian pipeline.

Extracted from the nonlinearity_studies package. This module contains exactly
the pieces needed to:
  1. Load FITS extensions.
  2. Row/column pedestal-subtract each extension (with an on-disk cache).
  3. Fit the zero/one-electron peaks with a double Gaussian (noise/gain/pedestal).
  4. Plot those double-Gaussian fits in ADU and/or electron units.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import biweight_location, biweight_midvariance
from scipy.optimize import curve_fit
from scipy.signal import find_peaks as scipy_find_peaks

from pathlib import Path
from tqdm import tqdm

#---------------- Curves ----------------------------

def double_gauss(x, s0, m0, s1, m1, N0, N1):
    return N0 * np.exp(-(x-m0)**2/(2*s0**2)) + N1 * np.exp(-(x-m1)**2/(2*s1**2))

#---------------- (0) Convert to electrons ----------------------------

def convert_to_electrons(data, pedestal, gain, flatten=True):
    if flatten:
        data = np.array(data).flatten()
    data_electrons = (data - pedestal) / gain  # Subtract pedestal (mean ADU of zero electron peak) and divide by gain
    return data_electrons

#---------------- Histogram / peak helpers ----------------------------

def _smooth_counts(counts, window=5):
    if len(counts) < 3 or window <= 1:
        return counts.astype(float)

    window = min(window, len(counts))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return counts.astype(float)

    kernel = np.ones(window) / window
    return np.convolve(counts, kernel, mode='same')

def _make_histogram(data, hist_range, n, max_bins=4000):
    left, right = hist_range
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        raise ValueError(f'Invalid histogram range: {hist_range}')

    nbins = min(max_bins, max(50, int(n * (right - left))))
    data_window = data[(data > left) & (data < right)]
    counts, edges = np.histogram(data_window, bins=nbins, range=(left, right))
    centers = 0.5 * (edges[:-1] + edges[1:])

    if len(centers) < 2:
        raise ValueError(f'Could not make a useful histogram for range {hist_range}')

    return data_window, counts, edges, centers

def _estimate_peak_width(centers, counts, peak_index):
    smooth_counts = _smooth_counts(counts)
    bin_width = centers[1] - centers[0]
    peak_height = smooth_counts[peak_index]

    if peak_height <= 0:
        return bin_width

    half_max = 0.5 * peak_height

    left_index = peak_index
    while left_index > 0 and smooth_counts[left_index] > half_max:
        left_index -= 1

    right_index = peak_index
    while right_index < len(smooth_counts) - 1 and smooth_counts[right_index] > half_max:
        right_index += 1

    if right_index <= left_index:
        return bin_width

    return max((centers[right_index] - centers[left_index]) / 2.355, bin_width)

def _clip_to_bounds(values, bounds):
    low, high = [np.array(bound, dtype=float) for bound in bounds]
    values = np.array(values, dtype=float)
    eps = np.maximum(1e-12, 1e-9 * np.maximum(1, np.abs(high - low)))
    return np.minimum(np.maximum(values, low + eps), high - eps)

def _auto_zero_one_setup(data, zero_one_test_range, n):
    use_auto_range = (
        zero_one_test_range is None
        or (isinstance(zero_one_test_range, str) and zero_one_test_range in ('auto', 'default'))
    )

    if use_auto_range:
        # Anchor the window on robust statistics. Using a low percentile for the
        # left edge (e.g. the 0.1th) lets a heavy outlier tail -- such as the
        # large +/-1e4 ADU artefact pixels in averaged images -- drag range_left
        # thousands of ADU away from the zero peak, blowing the histogram bin
        # width up to several ADU. The zero-peak width is then overestimated,
        # which later suppresses a real one-electron peak as "too close" to the
        # zero peak. Instead centre on the median (the zero peak) and mirror the
        # 80th-percentile half-width about it, with a robust-sigma floor so the
        # window always spans the zero peak.
        median = float(np.median(data))
        mad = float(np.median(np.abs(data - median)))
        robust_sigma = 1.4826 * mad
        half_width = max(np.percentile(data, 80) - median, 3 * robust_sigma)
        range_left, range_right = median - half_width, median + half_width

        if not np.isfinite(range_left) or not np.isfinite(range_right) or range_right <= range_left:
            range_left = np.min(data)
            range_right = np.max(data)
    else:
        range_left, range_right = zero_one_test_range

    _, counts_test, edges_test, centers_test = _make_histogram(data, (range_left, range_right), n)

    if max(counts_test) == 0:
        raise ValueError(
            f'No data found in zero-one test range {(range_left, range_right)}. '
            'Use zero_one_test_range="auto" or choose a range around the pedestal.'
        )

    smooth_test = _smooth_counts(counts_test)
    zero_peak_index = np.argmax(smooth_test)
    zero_peak_charge = centers_test[zero_peak_index]
    zero_peak_width = _estimate_peak_width(centers_test, counts_test, zero_peak_index)
    bin_width = centers_test[1] - centers_test[0]
    zero_peak_width = max(zero_peak_width, bin_width)

    search_left = zero_peak_charge - 5 * zero_peak_width
    search_right = zero_peak_charge + max(20 * zero_peak_width, 2 * (range_right - zero_peak_charge))

    data_high = np.percentile(data, 99)
    if np.isfinite(data_high):
        search_right = min(search_right, data_high)

    if search_right <= zero_peak_charge:
        search_right = zero_peak_charge + 20 * zero_peak_width

    _, search_counts, search_edges, search_centers = _make_histogram(data, (search_left, search_right), n)
    smooth_search = _smooth_counts(search_counts, window=9)
    search_bin_width = search_centers[1] - search_centers[0]
    peak_distance = max(1, int(2 * zero_peak_width / search_bin_width))
    peak_prominence = max(5, 0.002 * max(smooth_search))
    peak_indices, _ = scipy_find_peaks(
        smooth_search,
        prominence=peak_prominence,
        distance=peak_distance,
    )

    one_peak_min_charge = zero_peak_charge + max(zero_peak_width, 0.2)
    right_peak_indices = [
        peak_index for peak_index in peak_indices
        if search_centers[peak_index] > one_peak_min_charge
    ]

    if right_peak_indices:
        found_one_peak = True
        one_peak_index = max(right_peak_indices, key=lambda peak_index: smooth_search[peak_index])
        one_peak_charge = search_centers[one_peak_index]
        one_peak_height = smooth_search[one_peak_index]
    else:
        found_one_peak = False
        one_peak_charge = zero_peak_charge + 4 * zero_peak_width
        one_peak_height = 0.1 * smooth_search[np.argmin(np.abs(search_centers - zero_peak_charge))]

    gain_guess = max(one_peak_charge - zero_peak_charge, 4 * zero_peak_width)

    zero_one_left = zero_peak_charge - max(4 * zero_peak_width, 0.5 * gain_guess)
    zero_one_right = zero_peak_charge + max(1.8 * gain_guess, 8 * zero_peak_width)
    zero_one_range = [zero_one_left, zero_one_right]

    _, zero_one_counts, zero_one_edges, zero_one_centers = _make_histogram(data, zero_one_range, n, max_bins=2000)
    max_zero_one_counts = max(zero_one_counts)

    if max_zero_one_counts == 0:
        raise ValueError(f'No data found in inferred zero-one range {zero_one_range}')

    fit_left, fit_right = zero_one_range
    m0_margin = max(2 * zero_peak_width, 0.3 * gain_guess)
    if found_one_peak:
        m1_margin = max(0.02 * zero_peak_width, 0.5 * (zero_one_centers[1] - zero_one_centers[0]))
        m1_low = max(
            zero_peak_charge + max(zero_peak_width, 0.2 * gain_guess),
            one_peak_charge - m1_margin,
        )
        m1_high = min(
            fit_right,
            zero_peak_charge + max(1.7 * gain_guess, 8 * zero_peak_width),
            one_peak_charge + m1_margin,
        )
    else:
        m1_low = zero_peak_charge + max(zero_peak_width, 0.2 * gain_guess)
        m1_high = min(fit_right, zero_peak_charge + max(1.7 * gain_guess, 8 * zero_peak_width))

    if m1_high <= m1_low:
        m1_high = fit_right

    fit_bounds_low = [
        max((zero_one_centers[1] - zero_one_centers[0]) / 10, 1e-8),
        max(fit_left, zero_peak_charge - m0_margin),
        max((zero_one_centers[1] - zero_one_centers[0]) / 10, 1e-8),
        m1_low,
        0,
        0,
    ]
    fit_bounds_high = [
        max(gain_guess, 4 * zero_peak_width),
        min(fit_right, zero_peak_charge + m0_margin),
        max(1.5 * gain_guess, 6 * zero_peak_width),
        m1_high,
        2 * max_zero_one_counts,
        2 * max_zero_one_counts,
    ]

    fit_bounds = (fit_bounds_low, fit_bounds_high)

    one_peak_bin = np.argmin(np.abs(zero_one_centers - one_peak_charge))
    one_peak_height = max(one_peak_height, zero_one_counts[one_peak_bin], 0.05 * max_zero_one_counts)
    p0 = [
        zero_peak_width,
        zero_peak_charge,
        max(1.5 * zero_peak_width, gain_guess / 3),
        one_peak_charge,
        max_zero_one_counts,
        one_peak_height,
    ]
    p0 = _clip_to_bounds(p0, fit_bounds)

    return zero_one_counts, zero_one_edges, p0, fit_bounds, zero_one_range

#---------------- (1) Calculate noise / gain ----------------------------

def calculate_noise_gain(data, zero_one_test_range='auto', n=100, fit_bounds='default'):

    data = np.array(data).flatten()
    data = data[np.isfinite(data)]

    if data.size == 0:
        raise ValueError('Input data contains no finite values')

    zero_one_counts, zero_one_edges, p0, auto_fit_bounds, zero_one_range = _auto_zero_one_setup(
        data,
        zero_one_test_range,
        n,
    )
    zero_one_centers = 0.5 * (zero_one_edges[:-1] + zero_one_edges[1:])

    if isinstance(fit_bounds, str) and fit_bounds in ('default', 'auto'):
        fit_bounds = auto_fit_bounds
    else:
        p0 = _clip_to_bounds(p0, fit_bounds)

    popt, pcov = curve_fit(double_gauss, zero_one_centers, zero_one_counts, p0=p0,
                           maxfev=20000, bounds=fit_bounds)
    
    # Extract pedestal, noise, gain, and rest of double gaussian coefficients from curve fit
    pedestal=tuple(popt)[1] # Pedestal is mean of zero electron peak
    noise=tuple(popt)[0] # Noise is standard deviation of zero electron peak 
    gain=tuple(popt)[3]-tuple(popt)[1] # Gain is difference between mean of one and zero electron peaks
    return zero_one_counts, zero_one_edges, pedestal, noise, gain, popt, zero_one_range

#---------------- (2) Pedestal subtraction ----------------------------

def pedestal_subtract(data, n_std_to_mask, axis='row', use_biweight_loc=True,
                      use_biweight_midvar=True, max_iter=5, tol=0.01,
                      verbose=False, label='', overscan_cols=None):
    """
    overscan_cols : tuple(int, int) or None
        When set to ``(c0, c1)``, the per-row pedestal is *estimated* only from
        columns ``c0:c1`` (a Python half-open slice, e.g. the serial overscan)
        but *subtracted* from the full row. This only affects the per-row step
        (``'row'``, and the row half of ``'row_then_col'`` / ``'col_then_row'``);
        the per-column step always uses every row, since restricting columns
        there would not produce a pedestal for the columns outside the slice.
    """

    data = np.array(data, dtype=float)
    log_prefix = f'  [pedsub] {label} ' if label else '  [pedsub] '

    def _loc(arr, ax):
        if use_biweight_loc:
            return biweight_location(arr, axis=ax, ignore_nan=True)
        return np.nanmean(arr, axis=ax)

    def _scale(arr, ax):
        if use_biweight_midvar:
            return np.sqrt(biweight_midvariance(arr, axis=ax, ignore_nan=True))  # returns variance, not std
        return np.nanstd(arr, axis=ax)

    def subtract_along(arr, ax, ax_name, sample=None):
        # ax=1 subtracts a per-row pedestal; ax=0 subtracts a per-column pedestal.
        #
        # `sample` (when given) is the sub-array the pedestal is ESTIMATED from --
        # e.g. just the overscan columns -- while the resulting per-line pedestal is
        # still SUBTRACTED from the full `arr`. It must share `arr`'s length along the
        # output axis (the complement of `ax`) so the estimate broadcasts back over the
        # full frame; for ax=1 that means the same number of rows. When None, the
        # estimate is drawn from `arr` itself (the original behaviour).
        #
        # Iteratively sigma-clip to the zero-peak core: each pass recomputes BOTH the
        # location and the scale from the surviving (clipped) pixels, so the mask width
        # converges to the zero-peak width rather than the inflated width of the full
        # (multi-peak) distribution. A single pass estimates the scale from the whole
        # line, which on noisy images (wide zero peak overlapping the one-electron peak)
        # leaves one-electron pixels inside the mask. Those sit at positive charge and
        # drag the pedestal high, so subtracting it over-corrects and pushes the zero
        # peak negative (the ~-0.2 ADU offset). Re-estimating the scale from the clipped
        # pixels peels that contamination off over a few passes.
        #
        # Early stop on the MEDIAN per-line shift: the bulk of lines converge in a few
        # passes, but a handful of sparse lines keep jittering by noise forever, so the
        # mask never repeats exactly and the max shift never settles. The median ignores
        # that thin tail and reflects when the pedestal has actually stabilized. Capped
        # at max_iter for the (noisy, overlapping) lines that need the full budget.
        est = arr if sample is None else sample
        center = _loc(est, ax)
        sigma = _scale(est, ax)
        shift = np.inf
        n_iter = 0
        for _ in range(max_iter):
            n_iter += 1
            center_b = np.expand_dims(center, axis=ax)
            sigma_b = np.expand_dims(sigma, axis=ax)
            mask = np.abs(est - center_b) <= n_std_to_mask * sigma_b
            masked = np.where(mask, est, np.nan)

            new_center = _loc(masked, ax)
            sigma = _scale(masked, ax)
            shift = np.nanmedian(np.abs(new_center - center))
            center = new_center
            if verbose:
                print(f'{log_prefix}{ax_name}: iter {n_iter}/{max_iter}  '
                      f'kept {np.mean(mask) * 100:5.1f}%  '
                      f'median |Δpedestal| = {shift:.2e} ADU')
            if not np.isfinite(shift) or shift <= tol:
                break

        if verbose:
            stopped = 'converged' if (np.isfinite(shift) and shift <= tol) else f'reached max_iter={max_iter}'
            print(f'{log_prefix}{ax_name}: {stopped} after {n_iter} iteration(s)')

        return arr - np.expand_dims(center, axis=ax)

    def _row_sample(arr):
        # Sub-array (overscan columns) the per-row pedestal is estimated from.
        if overscan_cols is None:
            return None
        c0, c1 = overscan_cols
        return arr[:, c0:c1]

    if axis == 'row':
        return subtract_along(data, 1, 'row', sample=_row_sample(data))
    elif axis in ('column', 'col'):
        # overscan_cols does not apply to a per-column pedestal (see docstring).
        return subtract_along(data, 0, 'col')
    elif axis == 'row_then_col':
        row_sub = subtract_along(data, 1, 'row', sample=_row_sample(data))
        return subtract_along(row_sub, 0, 'col')
    elif axis == 'col_then_row':
        col_sub = subtract_along(data, 0, 'col')
        return subtract_along(col_sub, 1, 'row', sample=_row_sample(col_sub))

    return data

# Bump when the pedestal_subtract algorithm changes so existing caches (whose axis/
# n_std/biweight params still match) are invalidated rather than silently reused.
#   1 = single-pass clip;  2 = iterative sigma-clip to the zero-peak core

_PEDSUB_ALGO_VERSION = 2

_PEDSUB_HEADER_KEYS = ('PEDSUB_A', 'PEDSUB_N', 'PEDSUB_L', 'PEDSUB_V', 'PEDSUB_R', 'PEDSUB_I', 'PEDSUB_O')


def _overscan_range_token(overscan_cols):
    """Token for a single extension's overscan range.

    ``None`` -> 'f' (estimate that extension from the full frame). Otherwise a
    ``(c0, c1)`` slice, with endpoints that may be negative (counted from the right)
    or ``None`` (open-ended slice), encoded so the token is filename- and
    header-string-safe.
    """
    if overscan_cols is None:
        return 'f'
    c0, c1 = overscan_cols

    def _fmt(v):
        if v is None:
            return 'e'                 # open-ended slice endpoint
        return str(int(v)).replace('-', 'm')   # 'm' keeps negatives filename-safe

    return f'{_fmt(c0)}t{_fmt(c1)}'


def _overscan_key(overscan_cols_per_ext):
    """Filename/header-safe token for the per-extension overscan configuration.

    Takes a list with one entry per extension, each either ``None`` (estimate that
    extension's pedestal from the full frame) or a ``(c0, c1)`` column slice. Returns
    'none' when no extension uses overscan-only estimation, so runs that touch no
    extension share a cache with the plain full-frame result.
    """
    if not overscan_cols_per_ext or all(o is None for o in overscan_cols_per_ext):
        return 'none'
    return '-'.join(_overscan_range_token(o) for o in overscan_cols_per_ext)


def _normalize_overscan_cols_ext(overscan_cols, n_ext):
    """Normalize the ``overscan_cols`` argument into a length-``n_ext`` per-extension list.

    Accepts ``None`` (no extension uses overscan), a single ``(c0, c1)`` pair (applied
    to every extension), or an explicit length-``n_ext`` list whose entries are each
    ``None`` or a ``(c0, c1)`` pair.
    """
    if overscan_cols is None:
        return [None] * n_ext
    # Explicit per-extension list (checked first so a 2-extension file isn't mistaken
    # for a single range): every entry must be None or a 2-sequence.
    if isinstance(overscan_cols, list) and len(overscan_cols) == n_ext and all(
            o is None or (isinstance(o, (list, tuple)) and len(o) == 2) for o in overscan_cols):
        return [tuple(o) if o is not None else None for o in overscan_cols]
    # A single (c0, c1) range -> apply to every extension.
    if isinstance(overscan_cols, (list, tuple)) and len(overscan_cols) == 2 and all(
            v is None or isinstance(v, (int, np.integer)) for v in overscan_cols):
        return [tuple(overscan_cols)] * n_ext
    raise ValueError(
        f"overscan_cols must be None, a (c0, c1) pair, or a length-{n_ext} per-extension "
        f"list of None/(c0, c1); got {overscan_cols!r}")


def _pedsub_param_key(axis, n_std_to_mask, use_biweight_loc, use_biweight_midvar,
                      max_iter, overscan_cols=None):
    """Compact filename-safe key for the params that define a distinct pedsub result.

    Mirrors the fields checked by ``_pedsub_header_matches`` so that every distinct
    parameter combination maps to its own cache file (e.g. row vs column live in
    separate files and don't overwrite each other).
    """
    return (f"{axis}_n{float(n_std_to_mask):g}"
            f"_l{int(bool(use_biweight_loc))}_v{int(bool(use_biweight_midvar))}"
            f"_i{int(max_iter)}_o{_overscan_key(overscan_cols)}_a{_PEDSUB_ALGO_VERSION}")


def _pedsub_cache_path(source_path, cache_dir=None, *, axis=None, n_std_to_mask=None,
                       use_biweight_loc=True, use_biweight_midvar=True, max_iter=5,
                       overscan_cols=None):
    source = Path(source_path)
    base = cache_dir if cache_dir is not None else source.parent
    base = Path(base)
    if cache_dir is not None:
        base.mkdir(parents=True, exist_ok=True)
    if axis is None:
        # Legacy single-slot path (no params supplied).
        return base / f'{source.stem}.pedsub.fits'
    key = _pedsub_param_key(axis, n_std_to_mask, use_biweight_loc,
                            use_biweight_midvar, max_iter, overscan_cols)
    return base / f'{source.stem}.pedsub.{key}.fits'

def _pedsub_header_matches(header, axis, n_std_to_mask, use_biweight_loc,
                           use_biweight_midvar, max_iter, overscan_cols=None):
    if not all(k in header for k in _PEDSUB_HEADER_KEYS):
        return False
    return (
        header['PEDSUB_A'] == axis
        and float(header['PEDSUB_N']) == float(n_std_to_mask)
        and bool(header['PEDSUB_L']) == bool(use_biweight_loc)
        and bool(header['PEDSUB_V']) == bool(use_biweight_midvar)
        and int(header['PEDSUB_R']) == _PEDSUB_ALGO_VERSION
        and int(header['PEDSUB_I']) == int(max_iter)
        and str(header['PEDSUB_O']) == _overscan_key(overscan_cols)
    )

def pedestal_subtract_ext_cached(data_ext, source_path, n_std_to_mask, axis='row',
                                 use_biweight_loc=True, use_biweight_midvar=True,
                                 max_iter=5, cache_dir=None, force=False, verbose=True,
                                 overscan_cols=None):
    """Pedestal-subtract each extension, caching the result to a FITS file next to the source.

    Each distinct parameter combination (axis, n_std_to_mask, biweight flags, max_iter,
    algorithm version) is cached to its own file, so switching e.g. axis from 'row' to
    'column' and back reuses each previously computed result instead of overwriting it.

    On rerun, if the matching cache exists and its header params agree with the requested
    params, the cached arrays are loaded instead of recomputing. Pass force=True to bypass
    the cache.

    ``overscan_cols`` may be None (no extension uses overscan-only estimation), a single
    ``(c0, c1)`` pair applied to every extension, or a per-extension list of None/(c0, c1)
    so individual extensions can be estimated from their overscan columns or the full frame.
    """
    # Resolve overscan to one entry per extension up front, so the cache identity and the
    # per-extension calls below agree.
    overscan_cols_ext = _normalize_overscan_cols_ext(overscan_cols, len(data_ext))

    cache_path = _pedsub_cache_path(source_path, cache_dir, axis=axis,
                                    n_std_to_mask=n_std_to_mask,
                                    use_biweight_loc=use_biweight_loc,
                                    use_biweight_midvar=use_biweight_midvar,
                                    max_iter=max_iter, overscan_cols=overscan_cols_ext)

    if not force and cache_path.exists():
        with fits.open(str(cache_path)) as hdul:
            if _pedsub_header_matches(hdul[0].header, axis, n_std_to_mask,
                                       use_biweight_loc, use_biweight_midvar, max_iter,
                                       overscan_cols_ext):
                if verbose:
                    print(f'Loading cached pedestal-subtracted data from {cache_path}')
                return [hdul[i].data.copy() for i in range(1, len(hdul))]
            elif verbose:
                print(f'Cached params at {cache_path} differ from current; recomputing.')

    # Progress feedback stays visible regardless of verbose; only the textual
    # cache messages (load/recompute/save) are gated by verbose. When verbose, each
    # extension prints its own per-iteration convergence trace, so the tqdm progress
    # bar (which would interleave with those prints) is dropped in favour of the trace.
    print('Computing pedestal subtraction...')
    iterable = data_ext if verbose else tqdm(data_ext, desc='Pedestal subtraction', unit='ext')
    pedsub_data_ext = [
        pedestal_subtract(data, n_std_to_mask=n_std_to_mask, axis=axis,
                          use_biweight_loc=use_biweight_loc, use_biweight_midvar=use_biweight_midvar,
                          max_iter=max_iter, verbose=verbose, label=f'EXT {i + 1}',
                          overscan_cols=overscan_cols_ext[i])
        for i, data in enumerate(iterable)
    ]

    primary = fits.PrimaryHDU()
    primary.header['PEDSUB_A'] = (axis, 'pedestal subtraction axis')
    primary.header['PEDSUB_N'] = (float(n_std_to_mask), 'n_std_to_mask')
    primary.header['PEDSUB_L'] = (bool(use_biweight_loc), 'use biweight location')
    primary.header['PEDSUB_V'] = (bool(use_biweight_midvar), 'use biweight midvariance')
    primary.header['PEDSUB_R'] = (_PEDSUB_ALGO_VERSION, 'pedestal subtraction algorithm version')
    primary.header['PEDSUB_I'] = (int(max_iter), 'pedestal subtraction max iterations')
    primary.header['PEDSUB_O'] = (_overscan_key(overscan_cols_ext), 'per-ext overscan cols for pedestal estimate')
    primary.header['SRC_FITS'] = (str(source_path)[-68:], 'source FITS file (truncated)')
    hdul_out = fits.HDUList([primary] + [fits.ImageHDU(data=arr) for arr in pedsub_data_ext])
    hdul_out.writeto(str(cache_path), overwrite=True)
    if verbose:
        print(f'Saved pedestal-subtracted cache to {cache_path}')

    return pedsub_data_ext

#---------------- Plotting: zero-one peaks ----------------------------

def _finish_fig(show_plots):
    """Display the current figure interactively (blocking), or close it when not showing.

    plt.show() displays every figure still registered with pyplot, so the current
    figure must be closed once the user dismisses its window; otherwise the next
    figure's plt.show() re-raises this one too and interactive GUI backends (notably
    macOS) pop up duplicates. Closing right after the blocking show keeps figures
    appearing one at a time with a clean registry. When not showing, the figure is
    closed immediately to free memory.
    """
    if show_plots:
        plt.show()
    plt.close()

def _zero_one_ylim(ylim, yscale, counts, fit_y):
    """Per-extension (low, high) for a zero-one subplot, or None to leave autoscaling.

    `counts`/`fit_y` are the bar heights and fitted curve for the extension and drive the
    autoscale ('none'/None) case, so callers can share a y-axis across extensions by taking
    the minimum low and maximum high of every extension's returned range.
    """
    if isinstance(ylim, str) and ylim == 'default':
        if yscale == 'log':
            return 0, np.max(counts) * 1e4
        if yscale == 'linear':
            return 0, np.max(counts) + 2.5e4
        return None
    if ylim is None or ylim == 'none':
        if yscale != 'linear':
            return None  # let matplotlib pick a sensible (e.g. log) range per axis
        return 0, max(np.max(counts), np.max(fit_y)) * 1.05
    return ylim[0], ylim[1]

def _fit_double_gauss_electrons(centers_e, counts_e, double_gauss_popt, pedestal, gain, maxfev=5000):
    """Fit the zero/one peak histogram in electron units, seeded from the converged ADU fit.

    The ADU fit already located both peaks, so converting it to electron units gives a
    physically anchored initial guess (mu0 == 0, mu1 == 1 by construction, since
    pedestal == popt[1] and gain == popt[3] - popt[1]) instead of refitting blind with
    a fixed p0 of all-ones. Bounds are derived from that guess rather than hardcoded, so
    the fit cannot collapse one Gaussian into a delta spike on a noise bump or slide the
    one-electron peak onto the pedestal -- the failure mode seen on low-statistics single
    images.
    """
    s0, m0, s1, m1 = (double_gauss_popt[0], double_gauss_popt[1],
                      double_gauss_popt[2], double_gauss_popt[3])
    s0_e = max(s0 / gain, 1e-3)
    s1_e = max(s1 / gain, 1e-3)
    m0_e = (m0 - pedestal) / gain   # 0 by construction
    m1_e = (m1 - pedestal) / gain   # 1 by construction

    N0_0 = max(counts_e[np.argmin(np.abs(centers_e - m0_e))], 1.0)
    N1_0 = max(counts_e[np.argmin(np.abs(centers_e - m1_e))], 1.0)
    cmax = max(np.max(counts_e), 1.0)

    p0 = [s0_e, m0_e, s1_e, m1_e, N0_0, N1_0]
    bounds = (
        [0.2 * s0_e, m0_e - 0.25, 0.2 * s1_e, m1_e - 0.3, 0, 0],
        [5.0 * s0_e, m0_e + 0.25, 5.0 * s1_e, m1_e + 0.3, 5 * cmax, 5 * cmax],
    )
    p0 = _clip_to_bounds(p0, bounds)
    return curve_fit(double_gauss, centers_e, counts_e, p0=p0, bounds=bounds, maxfev=maxfev)

def plot_zero_one_peaks(data_ext,
                        zero_one_counts_ext,
                        zero_one_edges_ext,
                        pedestals, 
                        gains, 
                        double_gauss_popts, 
                        zero_one_ranges,
                        individual_figsize=(7,5),
                        subplots_figsize=(13,9),
                        xlim='default',
                        ylim='default',
                        ylim_electrons='default',
                        suptitle='Double-Gaussian Fit to Zero-One Electron Peaks',
                        additional_title='',
                        nimages=10,
                        fontsize=9.5,
                        yscale='linear',
                        n=100,
                        do_convert_to_electrons=False,
                        plot_individual=False,
                        plot_together=True,
                        do_plot_adu=True,
                        sharex=True,
                        sharey=True,
                        show_titles=True,
                        save_plots=False,
                        show_plots=True,
                        fig_path='./', file='zero_one_peaks',
                        dpi=350):

    fig_path = Path(fig_path)
    if file != 'zero_one_peaks':
        base_name = file[:-5] + '_zero_one_peaks'
    else:
        base_name = file
    fig_name = fig_path / base_name

    if plot_individual:
        for ext, data in (enumerate(data_ext) if do_plot_adu else []):
            data = np.array(data).flatten()

            zero_one_counts=zero_one_counts_ext[ext]
            zero_one_edges=zero_one_edges_ext[ext]
            pedestal=pedestals[ext]
            gain=gains[ext]
            double_gauss_popt=double_gauss_popts[ext]
            zero_one_range=zero_one_ranges[ext]

            fig, ax = plt.subplots(1, 1, figsize=individual_figsize, constrained_layout=True)
            if show_titles:
                ax.set_title(f'{additional_title}{suptitle} (Nimages = {nimages}): EXT {ext + 1}', fontsize=12, pad=10)
            ax.set_xlabel('Charge (ADU)')
            ax.set_ylabel('N')

            double_gauss_coeff = tuple(double_gauss_popt)+(gain,)
            data_window = data[(data > zero_one_range[0]) & (data < zero_one_range[1])]
            nbins=int(n*(zero_one_range[1] - zero_one_range[0]))

            bin_width = zero_one_edges[1] - zero_one_edges[0]
            zero_one_centers = 0.5 * (zero_one_edges[:-1] + zero_one_edges[1:])

            if yscale=='log':
                zero_one_counts = np.maximum(zero_one_counts, 1) #need in order to prevent empty bars in histogram if there are any bins that have 0 counts (log(0) is infinite)
                ax.set_yscale('log')
            elif yscale!='linear':
                ax.set_yscale(yscale)
            ax.bar(zero_one_edges[:-1], zero_one_counts, edgecolor='none', linewidth=0, align='edge', width=np.diff(zero_one_edges))
            
            ax.plot(zero_one_centers, double_gauss(zero_one_centers, *double_gauss_popt), 'r',
                label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ = %5.3f, $\mu_1$ = %5.3f,'%double_gauss_coeff[0:4]
                +'\n'+'$N_0$ = %5.3f, $N_1$ = %5.3f, gain = %5.3f ADU/$e^{–}$'%double_gauss_coeff[4:])
            ax.legend(loc="upper right", fontsize=fontsize)
            
            if xlim=='default':
                ax.set_xlim(zero_one_range[0],zero_one_range[1])
            elif xlim!='none':
                ax.set_xlim(xlim)
            
            if ylim=='default':
                if yscale=='log':
                    ax.set_ylim(0, np.max(zero_one_counts) * 1e4)
                elif yscale=='linear':
                    ax.set_ylim(0, np.max(zero_one_counts) + 3e4)
            elif ylim!='none':
                ax.set_ylim(ylim)

            if save_plots:
                output_path = fig_name.with_stem(fig_name.stem + f'_EXT{ext+1}').with_suffix('.jpeg')
                plt.savefig(str(output_path), dpi=dpi)
                print(f'Saved plots to {output_path}')
            _finish_fig(show_plots)

        if do_convert_to_electrons:
            for ext, data in enumerate(data_ext):
                data=np.array(data).flatten()
                zero_one_range = zero_one_ranges[ext]
                pedestal = pedestals[ext]
                gain = gains[ext]

                data_window=data[(data > zero_one_range[0]) & (data < zero_one_range[1])]

                data_window_e = convert_to_electrons(data_window, pedestal, gain)
                zero_one_range_e = convert_to_electrons(zero_one_range, pedestal, gain)
                nbins = int(n * (zero_one_range_e[1] - zero_one_range_e[0]))

                zero_one_counts_e, zero_one_edges_e = np.histogram(data_window_e, bins=nbins, range=zero_one_range_e)
                zero_one_centers_e = 0.5 * (zero_one_edges_e[:-1] + zero_one_edges_e[1:])
                bin_width_e = zero_one_edges_e[1] - zero_one_edges_e[0]

                double_gauss_popt_e, _ = _fit_double_gauss_electrons(
                    zero_one_centers_e, zero_one_counts_e, double_gauss_popts[ext], pedestal, gain)

                fig, ax = plt.subplots(1, 1, figsize=individual_figsize, constrained_layout=True)
                if show_titles:
                    ax.set_title(rf'{additional_title}{suptitle} (Nimages = {nimages}): EXT {ext + 1}', fontsize=12, pad=10)

                if yscale=='log':
                    zero_one_counts_e = np.maximum(zero_one_counts_e, 1)
                    ax.set_yscale('log')
                elif yscale!='linear':
                    ax.set_yscale(yscale)

                ax.bar(zero_one_edges_e[:-1], zero_one_counts_e, align='edge', edgecolor='none', linewidth=0, width=np.diff(zero_one_edges_e))
                ax.set_xlabel(r'Charge ($e^–$)')
                ax.set_ylabel('N')
                ax.plot(zero_one_centers_e, double_gauss(zero_one_centers_e, *double_gauss_popt_e), 'r',
                    label=r'$\sigma_0$ = %5.3f $e^{–}$, $\mu_0$ = %5.3f $e^{–}$, $\sigma_1$ = %5.3f $e^{–}$, $\mu_1$ = %5.3f $e^{–}$,'%tuple(double_gauss_popt_e)[0:4]
                    +'\n'+'gain = %5.3f ADU/$e^{–}$'%gain)
                ax.legend(loc="upper right", fontsize=fontsize)

                if xlim=='default':
                    ax.set_xlim(zero_one_range_e[0], zero_one_range_e[1])
                elif xlim!='none':
                    ax.set_xlim(xlim)

                if ylim_electrons=='default':
                    if yscale=='log':
                        ax.set_ylim(0, np.max(zero_one_counts_e) * 1e4)
                    elif yscale=='linear':
                        ax.set_ylim(0, np.max(zero_one_counts_e) + 2.5e4)
                elif ylim_electrons!='none':
                    ax.set_ylim(ylim_electrons)

                if save_plots:
                    output_path = fig_name.with_stem(fig_name.stem + f'_electrons_EXT{ext+1}').with_suffix('.jpeg')
                    plt.savefig(str(output_path), dpi=dpi)
                    print(f'Saved plot to {output_path}')
                _finish_fig(show_plots)

    if plot_together:

        if do_plot_adu:
            fig, axs = plt.subplots(2, 2, figsize=subplots_figsize, constrained_layout=True, sharex=sharex, sharey=sharey)
            if show_titles:
                fig.suptitle(f'{additional_title}{suptitle} (Nimages = {nimages})')
            axs = axs.flatten()

            _ylims = []
            for ext, data in enumerate(data_ext):
                data = np.array(data).flatten()
                zero_one_counts=zero_one_counts_ext[ext]
                zero_one_edges=zero_one_edges_ext[ext]
                pedestal=pedestals[ext]
                gain=gains[ext]
                double_gauss_popt=double_gauss_popts[ext]
                zero_one_range=zero_one_ranges[ext]

                ax = axs[ext]
                double_gauss_coeff = tuple(double_gauss_popt)+(gain,)
                data_window = data[(data > zero_one_range[0]) & (data < zero_one_range[1])]
                nbins=int(n*(zero_one_range[1]-zero_one_range[0]))

                zero_one_centers = 0.5 * (zero_one_edges[:-1] + zero_one_edges[1:])
                bin_width = zero_one_edges[1] - zero_one_edges[0]

                if yscale=='log':
                    zero_one_counts = np.maximum(zero_one_counts, 1) #need in order to prevent empty bars in histogram if there are any bins that have 0 counts
                    ax.set_yscale('log')
                elif yscale!='linear':
                    ax.set_yscale(yscale)

                ax.bar(zero_one_centers, zero_one_counts, align='center', edgecolor='none', linewidth=0, width=bin_width)

                fit_y = double_gauss(zero_one_centers, *double_gauss_popt)
                ax.plot(zero_one_centers, fit_y, 'r',
                    label=r'$\sigma_0$ = %5.3f, $\mu_0$ = %5.3f, $\sigma_1$ = %5.3f, $\mu_1$ = %5.3f,'%double_gauss_coeff[0:4]
                    +'\n'+'$N_0$ = %5.3f, $N_1$ = %5.3f, gain = %5.3f ADU/$e^{–}$'%double_gauss_coeff[4:])

                ax.set_xlabel('Charge (ADU)')
                ax.set_ylabel('N')
                ax.set_title(f'EXT {ext + 1}')
                ax.legend(loc="upper right", fontsize=fontsize - 2)

                if xlim=='default':
                    ax.set_xlim(zero_one_range[0],zero_one_range[1])
                elif xlim!='none':
                    ax.set_xlim(xlim)

                lim = _zero_one_ylim(ylim, yscale, zero_one_counts, fit_y)
                _ylims.append(lim)
                if not sharey and lim is not None:
                    ax.set_ylim(lim)

            # With sharey=True every subplot shares one range; span every extension by
            # taking the minimum low and maximum high so no peak is clipped (the tallest
            # peak need not be in the same extension as the lowest floor).
            _valid = [lim for lim in _ylims if lim is not None]
            if sharey and _valid:
                axs[0].set_ylim(min(lo for lo, _ in _valid), max(hi for _, hi in _valid))

            for i in (0, 1):
                axs[i].set_xlabel('')
                axs[i].tick_params(labelbottom=True)
            for i in (1, 3):
                axs[i].set_ylabel('')
                axs[i].tick_params(labelleft=True)

            if save_plots:
                output_path = fig_name.with_suffix('.jpeg')
                plt.savefig(str(output_path), dpi=dpi)
                print(f'Saved plot to {output_path}')
            _finish_fig(show_plots)

        if do_convert_to_electrons:
            fig, axs = plt.subplots(2, 2, figsize=subplots_figsize, constrained_layout=True, sharex=sharex, sharey=sharey)
            if show_titles:
                fig.suptitle(rf'{additional_title}{suptitle} (Nimages = {nimages})')
            axs = axs.flatten()

            _ylims = []
            for ext, data in enumerate(data_ext):
                ax = axs[ext]

                data = np.array(data).flatten()
                zero_one_range = zero_one_ranges[ext]
                pedestal = pedestals[ext]
                gain = gains[ext]

                data_window = data[(data > zero_one_range[0]) & (data < zero_one_range[1])]

                data_window_e = convert_to_electrons(data_window, pedestal, gain)
                zero_one_range_e = convert_to_electrons(zero_one_range, pedestal, gain)
                nbins = int(n * (zero_one_range_e[1] - zero_one_range_e[0]))

                zero_one_counts_e, zero_one_edges_e = np.histogram(data_window_e, bins=nbins, range=zero_one_range_e)
                zero_one_centers_e = 0.5 * (zero_one_edges_e[:-1] + zero_one_edges_e[1:])
                bin_width_e = zero_one_edges_e[1] - zero_one_edges_e[0]

                zero_one_counts=zero_one_counts_ext[ext]
                zero_one_edges=zero_one_edges_ext[ext]

                double_gauss_popt_e, _ = _fit_double_gauss_electrons(
                    zero_one_centers_e, zero_one_counts_e, double_gauss_popts[ext], pedestal, gain)

                if yscale=='log':
                    zero_one_counts_e = np.maximum(zero_one_counts_e, 1) #need in order to prevent empty bars in histogram if there are any bins that have 0 counts
                    ax.set_yscale('log')

                elif yscale!='linear':
                    ax.set_yscale(yscale)

                ax.bar(zero_one_centers_e, zero_one_counts_e, align='center', edgecolor='none', linewidth=0, width=bin_width_e)

                ax.set_title(f'EXT {ext + 1}')
                fit_y_e = double_gauss(zero_one_centers_e, *double_gauss_popt_e)
                ax.plot(zero_one_centers_e, fit_y_e, 'r',
                    label=r'$\sigma_0$ = %5.3f $e^{–}$, $\mu_0$ = %5.3f $e^{–}$, $\sigma_1$ = %5.3f $e^{–}$, $\mu_1$ = %5.3f $e^{–}$,'%tuple(double_gauss_popt_e)[0:4]
                    +'\n'+'gain = %5.3f ADU/$e^{–}$'%gain)
                ax.legend(loc="upper right", fontsize=fontsize - 2)
                ax.set_xlabel(r'Charge ($e^–$)')
                ax.set_ylabel('N')

                if xlim=='default':
                    ax.set_xlim(zero_one_range_e[0], zero_one_range_e[1])
                elif xlim!='none':
                    ax.set_xlim(xlim)

                lim = _zero_one_ylim(ylim_electrons, yscale, zero_one_counts_e, fit_y_e)
                _ylims.append(lim)
                if not sharey and lim is not None:
                    ax.set_ylim(lim)

            _valid = [lim for lim in _ylims if lim is not None]
            if sharey and _valid:
                axs[0].set_ylim(min(lo for lo, _ in _valid), max(hi for _, hi in _valid))

            for i in (0, 1):
                axs[i].set_xlabel('')
                axs[i].tick_params(labelbottom=True)
            for i in (1, 3):
                axs[i].set_ylabel('')
                axs[i].tick_params(labelleft=True)

            if save_plots:
                output_path = fig_name.with_stem(fig_name.stem + '_electrons').with_suffix('.jpeg')
                plt.savefig(str(output_path), dpi=dpi)
                print(f'Saved plot to {output_path}')
            _finish_fig(show_plots)

#---------------- Load FITS ----------------------------

def get_fits(file_input):
    """
    Load FITS extensions from a file.

    Parameters
    ----------
    file_input : str or Path
        Path to the FITS file (absolute or relative to current working directory)

    Returns
    -------
    ext_charge : list
        List of data arrays from extensions 1–4
    """
    file_path = Path(file_input).resolve()
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"FITS file not found: {file_path}")

    # Load FITS file
    with fits.open(str(file_path)) as hdu_list:
        ext_charge = [hdu_list[i].data for i in range(1, 5)]

    return ext_charge

#---------------- Per-extension zero/one fitting ----------------------------

def _value_for_extension(value, ext, n_ext):
    if isinstance(value, (list, tuple)) and len(value) == n_ext:
        if all(isinstance(v, (list, tuple, np.ndarray)) and len(v) == 2 for v in value):
            return value[ext]
    return value

def get_zero_one_peaks_ext(data_ext,
                           n=100, fit_bounds='default', zero_one_test_range='auto'):
    zero_one_counts_ext = []
    zero_one_edges_ext = []
    pedestals = []
    gains = []
    double_gauss_popts = []
    zero_one_ranges = []
    for ext, data in enumerate(data_ext):

        zero_one_test_range_ext = _value_for_extension(zero_one_test_range, ext, len(data_ext))

        zero_one_counts, zero_one_edges, pedestal, noise, gain, double_gauss_popt, zero_one_range = calculate_noise_gain(
            data,
            zero_one_test_range=zero_one_test_range_ext,
            n=n,
            fit_bounds=fit_bounds,
        )
        zero_one_counts_ext.append(zero_one_counts)
        zero_one_edges_ext.append(zero_one_edges)
        pedestals.append(pedestal)
        gains.append(gain)
        double_gauss_popts.append(double_gauss_popt)
        zero_one_ranges.append(zero_one_range)

    return zero_one_counts_ext, zero_one_edges_ext, pedestals, gains, double_gauss_popts, zero_one_ranges
