"""
pedestal — split out of core.py.
"""
import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location
from astropy.stats import biweight_midvariance
from pathlib import Path
from tqdm import tqdm

_PEDSUB_ALGO_VERSION = 2


_PEDSUB_HEADER_KEYS = ('PEDSUB_A', 'PEDSUB_N', 'PEDSUB_L', 'PEDSUB_V', 'PEDSUB_R', 'PEDSUB_I', 'PEDSUB_O')


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
    # Default to a `cache/` subfolder beside the source so the pedsub FITS files
    # don't clutter the data directory; an explicit cache_dir overrides this.
    base = Path(cache_dir) if cache_dir is not None else source.parent / 'cache'
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
    """Pedestal-subtract each extension, caching the result to a FITS file in a ``cache/`` folder beside the source.

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
