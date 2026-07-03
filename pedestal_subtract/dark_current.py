"""
dark_current — split out of core.py.
"""
import numpy as np
from scipy.special import erf

from .constants import _DARK_CURRENT_COLUMN, _DARK_CURRENT_METHODS

def _resolve_dark_current_methods(method):
    """Methods to compute, from a 'count'/'integrate'/'weighted'/'all' selector, or a
    list/tuple of those (each entry may also be 'all'). Returned in stable canonical
    (``_DARK_CURRENT_METHODS``) order regardless of input order, so CSV columns stay
    deterministic."""
    items = (method,) if isinstance(method, str) else tuple(method)
    if not items:
        raise ValueError("dark current method must include at least one of 'count', "
                         "'integrate', 'weighted', or 'all'; got an empty selection")
    bad = [m for m in items if m not in _DARK_CURRENT_METHODS and m != 'all']
    if bad:
        raise ValueError("dark current method must be one of 'count', 'integrate', "
                         f"'weighted', or 'all'; got invalid value(s) {bad!r}")
    if 'all' in items:
        return _DARK_CURRENT_METHODS
    return tuple(m for m in _DARK_CURRENT_METHODS if m in items)


def _single_electron_count_window(data, pedestal, gain, sigma0_adu, mu1, count_center,
                                  nsigma=1.0):
    """Number of pixels within ``nsigma`` peak sigmas of the 1 e- charge (method 1).

    Counts every pixel whose charge lies within +/- ``nsigma * sigma0_adu`` (a multiple
    of the fitted peak width sigma) of the one-electron charge. The window is centred
    either on the *ideal* 1 e- (``pedestal + gain``, ``count_center='one_electron'``)
    or on the *fitted* one-electron mean ``mu1`` (``count_center='mu1'``).

    The half-width uses the single shared width sigma fit to both peaks; it is set by
    the bulk of the pixels (the zero peak) and so is well-determined even when there
    are only a handful of single-electron events, keeping the window width stable
    across extensions.

    The zero-peak tail is deliberately NOT subtracted: those pixels cannot be physically
    distinguished from genuine single-electron events, so this is an inclusive
    (upper-bound) count. A narrower window (``nsigma < 1``) cuts the zero-peak tail
    contamination that dominates at low dark current, at the cost of capturing a smaller
    fraction of the true 1 e- population.

    The +/- ``nsigma`` window holds only a fraction ``erf(nsigma / sqrt(2))`` of a
    Gaussian's area (0.6827 at nsigma = 1), so the raw count is divided by that fraction
    to estimate the *total* number of pixels under the one-electron Gaussian -- bringing
    this method into agreement with the 'integrate' and 'weighted' methods. (Now that
    both peaks share one width, the window width matches the one-electron peak exactly,
    so the correction is the simple Gaussian fraction.)

    Returns ``np.nan`` when the gain is undefined (no one-electron peak).
    """
    if not np.isfinite(gain):
        return np.nan
    center = mu1 if count_center == 'mu1' else pedestal + gain
    half = nsigma * sigma0_adu
    lo, hi = center - half, center + half
    flat = np.asarray(data).flatten()
    flat = flat[np.isfinite(flat)]
    count = float(np.count_nonzero((flat >= lo) & (flat <= hi)))
    return count / erf(nsigma / np.sqrt(2))


def _single_electron_count_integral(double_gauss_popt, zero_one_edges, gain):
    """Number of single-electron events from the area under the fitted 1 e- Gaussian (method 2).

    The fitted one-electron Gaussian is ``N1 * exp(-(x-m1)^2 / (2 s^2))`` in counts
    vs charge (ADU), where ``s`` is the shared peak width; its analytic area is
    ``N1 * s * sqrt(2*pi)`` (counts*ADU). Dividing by the histogram bin width converts
    that area to a pixel count. This is a conservative/upper-bound estimate and is
    sensitive to the quality of the fit.

    Returns ``np.nan`` when the gain is undefined (the one-electron amplitude has been
    zeroed upstream, so there is no trustworthy peak to integrate).
    """
    if not np.isfinite(gain):
        return np.nan
    s, N1 = double_gauss_popt[0], double_gauss_popt[4]
    bin_width = zero_one_edges[1] - zero_one_edges[0]
    if bin_width <= 0:
        return np.nan
    area = N1 * s * np.sqrt(2 * np.pi)  # counts * ADU
    return area / bin_width


def _single_electron_count_weighted(double_gauss_popt, gain):
    """One-electron peak's amplitude fraction of the double Gaussian (method 3).

    The "event count" is simply the one-electron amplitude fraction
    ``n_events = N1 / (N1 + N0)`` -- the share of the two-peak amplitude carried by the
    one-electron Gaussian. This is already a per-image-pixel quantity (the fraction of
    image pixels in the one-electron peak), so ``calculate_dark_current`` forms the rate
    by dividing only by the pixel binning (to reach a per-*physical*-pixel rate) and the
    exposure -- it is NOT divided by the pixel count, which would cancel in the ratio.

    Returns ``np.nan`` when the gain is undefined (the one-electron amplitude has been
    zeroed upstream, so there is no trustworthy peak) or when the amplitudes are
    non-positive (no fraction can be formed).
    """
    if not np.isfinite(gain):
        return np.nan
    N0, N1 = double_gauss_popt[3], double_gauss_popt[4]
    denom = N1 + N0
    if denom <= 0:
        return np.nan
    return N1 / denom


def calculate_dark_current(data_ext, pedestals, gains, double_gauss_popts,
                           zero_one_edges_ext, exposure_days, method='all',
                           count_center='one_electron', count_nsigma=1.0,
                           pixel_binning=1):
    """Per-extension dark current (electrons / physical pixel / day) by one or more methods.

    Dark current = (number of single-electron events) / (number of physical pixels) /
    (exposure time in days). The single-electron events are counted by:

    - ``'count'``: pixels within ``count_nsigma`` peak sigmas of the 1 e- charge,
      divided by the Gaussian fraction ``erf(count_nsigma/sqrt(2))`` to estimate the
      full one-electron count (``_single_electron_count_window``; ``count_center``
      selects the centre, ``count_nsigma`` the half-width). The half-width uses the
      single fitted peak width sigma, well-determined even at low statistics (see that
      function). Use ``count_nsigma < 1`` to shrink the window and reduce zero-peak tail
      contamination.
    - ``'integrate'``: area under the fitted one-electron Gaussian
      (``_single_electron_count_integral``; a conservative upper bound).
    - ``'weighted'``: the one-electron amplitude fraction ``N1 / (N1 + N0)``
      (``_single_electron_count_weighted``). This is already per image pixel, so its
      rate is ``N1/(N1+N0) / pixel_binning / exposure_days`` -- divided only by the
      binning and exposure, NOT by the pixel count (which cancels in the ratio).

    ``method`` may be a single ``'count'``/``'integrate'``/``'weighted'``/``'all'``
    selector, or a list/tuple of any combination of those (``'all'`` expands to every
    method wherever it appears) -- only the requested method(s) are computed.

    The detector is read out binned, so each image pixel sums ``pixel_binning``
    (= NPBIN * NSBIN) physical CCD pixels. The number of physical pixels is therefore
    each extension's array size times ``pixel_binning`` (the dark current is quoted per
    *physical* pixel). Since both the event count and the pixel count scale with the
    analysed area, the rate does not depend on how many rows/columns are kept. When the
    gain is undefined or the exposure time is non-positive/NaN, the result is NaN.

    Returns a list (one dict per extension) whose keys are the CSV column names in
    ``_DARK_CURRENT_COLUMN`` for the requested method(s).
    """
    methods = _resolve_dark_current_methods(method)
    valid_exposure = np.isfinite(exposure_days) and exposure_days > 0

    rows = []
    for ext, gain in enumerate(gains):
        data = data_ext[ext]
        n_pixels = np.asarray(data).size * pixel_binning
        pedestal = pedestals[ext]
        popt = double_gauss_popts[ext]
        sigma0_adu = popt[0]  # s: shared peak width (= the pedestal/zero-peak width)
        mu1 = popt[2]

        def _rate(n_events):
            if not (np.isfinite(n_events) and valid_exposure and n_pixels > 0):
                return np.nan
            return n_events / n_pixels / exposure_days

        def _weighted_rate(fraction):
            # The 'weighted' method's n_events is the per-image-pixel fraction
            # N1/(N1+N0), so it must NOT be divided by the pixel count again (that
            # cancels in the ratio). Converting to a per-*physical*-pixel/day rate only
            # needs the binning (physical pixels summed per image pixel) and exposure.
            if not (np.isfinite(fraction) and valid_exposure and pixel_binning > 0):
                return np.nan
            return fraction / pixel_binning / exposure_days

        row = {}
        if 'count' in methods:
            n_events = _single_electron_count_window(
                data, pedestal, gain, sigma0_adu, mu1, count_center, count_nsigma)
            row[_DARK_CURRENT_COLUMN['count']] = _rate(n_events)
        if 'integrate' in methods:
            n_events = _single_electron_count_integral(popt, zero_one_edges_ext[ext], gain)
            row[_DARK_CURRENT_COLUMN['integrate']] = _rate(n_events)
        if 'weighted' in methods:
            n_events = _single_electron_count_weighted(popt, gain)
            row[_DARK_CURRENT_COLUMN['weighted']] = _weighted_rate(n_events)
        rows.append(row)
    return rows
