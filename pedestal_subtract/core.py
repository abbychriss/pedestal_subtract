"""
Core analysis routines for the pedestal-subtraction + double-Gaussian pipeline.

Extracted from the nonlinearity_studies package. This module contains exactly
the pieces needed to:
  1. Load FITS extensions.
  2. Row/column pedestal-subtract each extension (with an on-disk cache).
  3. Fit the zero/one-electron peaks with a double Gaussian (noise/gain/pedestal).
  4. Plot those double-Gaussian fits in ADU and/or electron units.
"""

import csv

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import biweight_location, biweight_midvariance
from scipy.optimize import curve_fit
from scipy.signal import find_peaks as scipy_find_peaks
from scipy.special import erf

from pathlib import Path
from tqdm import tqdm

#---------------- Curves ----------------------------

def double_gauss(x, s, m0, m1, N0, N1):
    # Both the zero- and one-electron Gaussians share a single width ``s`` (the readout
    # noise): the peaks are physically the same noise distribution shifted by 1 e-, so
    # one sigma is fit to both. popt layout is (s, m0, m1, N0, N1).
    return N0 * np.exp(-(x-m0)**2/(2*s**2)) + N1 * np.exp(-(x-m1)**2/(2*s**2))

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

# Default bins-per-ADU for the internal peak-finding histograms (locating the
# zero/one peaks and estimating their widths). Kept independent of the user-facing
# `n` (which sets the fit-window bin count) so lowering `n` does not degrade peak
# detection; exposed as the `peakfind_density` parameter / `zero_one_peakfind_density`
# config key so it can be tuned separately for low-statistics images. Coarser bins
# (a low density) aggregate sparse single-electron hits into a detectable cluster,
# so the default is deliberately low. Note `_peakfind_bins` floors the histogram at
# 50 bins, so any density small enough to fall below that floor behaves identically.
_PEAKFIND_DENSITY = 10

# Physically allowed range for the gain (ADU/e-). The gain converts ADU to
# electrons and is always > 1, and here we further cap it: the one-electron peak
# must sit in [pedestal + _MIN_GAIN_ADU, pedestal + _MAX_GAIN_ADU]. This band is
# used both to constrain the fit's mu_1 and to reject implausible fits. Kept as
# module constants so they are easy to tune; promote to config options if needed.
_MIN_GAIN_ADU = 0.5
_MAX_GAIN_ADU = 1.5
# Extra headroom (ADU) the fit is allowed *above* the gain band's upper edge, so a
# feature beyond the cap is fit at its true (high) centre and then rejected, rather
# than pinning at 1.5 and passing. The lower edge stays hard (the gain is always
# > 1). Acceptance still uses the band [_MIN_GAIN_ADU, _MAX_GAIN_ADU].
_GAIN_FIT_MARGIN = 0.3

# Default figure sizes (inches), shared by every plot function so all the multi-panel
# (2x2) figures match and all the single-panel figures match. Overridable per call.
_SUBPLOTS_FIGSIZE = (13, 9)    # 2x2 grids (zero/one "together", dark current, charge-per-column)
_INDIVIDUAL_FIGSIZE = (8, 6)   # single-panel per-extension figures

def _peakfind_bins(hist_range, density=_PEAKFIND_DENSITY):
    left, right = hist_range
    return min(4000, max(50, int(density * (right - left))))

def _make_histogram(data, hist_range, n, max_bins=4000):
    left, right = hist_range
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        raise ValueError(f'Invalid histogram range: {hist_range}')

    # `n` is the desired number of bins spanning this range (not a per-ADU
    # density), so the fit window gets ~n bins regardless of its width in ADU.
    nbins = min(max_bins, max(10, int(n)))
    data_window = data[(data > left) & (data < right)]
    counts, edges = np.histogram(data_window, bins=nbins, range=(left, right))
    centers = 0.5 * (edges[:-1] + edges[1:])

    if len(centers) < 2:
        raise ValueError(f'Could not make a useful histogram for range {hist_range}')

    return data_window, counts, edges, centers

def _bar_heights(counts, yscale):
    """Bar heights for plotting.

    On a log y-axis, zero-count bins are set to NaN so matplotlib draws nothing
    there -- the bins are genuinely empty rather than clamped up to 1, which
    previously produced a solid rectangular block along the bottom of images
    dominated by empty (zero-electron) bins.
    """
    counts = np.asarray(counts, dtype=float)
    if yscale == 'log':
        heights = counts.copy()
        heights[heights <= 0] = np.nan
        return heights
    return counts

def _fit_curve_x(lo, hi, n_bins, points_per_bin=20, floor=500):
    """x-grid for drawing the fitted curve, sampled at `points_per_bin` points per
    histogram bin so the curve stays smooth when zoomed in -- the point density
    per unit charge is constant regardless of how wide the window is scaled."""
    n = max(floor, int(points_per_bin) * int(n_bins))
    return np.linspace(lo, hi, n)

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

def _auto_zero_one_setup(data, zero_one_test_range, n, window_left_scale=1.0, window_right_scale=1.0,
                         peakfind_density=_PEAKFIND_DENSITY, gain_seed=None):
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

    test_range = (range_left, range_right)
    _, counts_test, edges_test, centers_test = _make_histogram(
        data, test_range, _peakfind_bins(test_range, peakfind_density))

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

    one_lo = zero_peak_charge + _MIN_GAIN_ADU
    one_hi = zero_peak_charge + _MAX_GAIN_ADU
    search_left = zero_peak_charge - 5 * zero_peak_width
    search_right = one_hi + 3 * zero_peak_width

    search_range = (search_left, search_right)
    _, search_counts, search_edges, search_centers = _make_histogram(
        data, search_range, _peakfind_bins(search_range, peakfind_density))
    search_bin_width = search_centers[1] - search_centers[0]
    # Smooth on the scale of the zero-peak width rather than re-binning coarsely.
    # In a low-statistics image the single-electron hits are sparse 1-count bins,
    # so on the raw fine histogram they never form a bump; averaging over ~one
    # zero-peak width lets a real cluster aggregate into a detectable peak while a
    # smooth monotonic tail stays bump-free. With coarse bins (low peakfind_density)
    # the bins already aggregate, so keep the smoothing window small -- a large
    # floor would wash a narrow one-electron bump (especially a low-gain one) into a
    # flat plateau that find_peaks can no longer locate.
    smooth_window = max(3, int(round(zero_peak_width / search_bin_width)))
    smooth_search = _smooth_counts(search_counts, window=smooth_window)
    peak_distance = max(1, int(2 * zero_peak_width / search_bin_width))
    # Reference the prominence to the Poisson noise of the *valley* level (a low
    # percentile of the tail counts), not the median: the median is inflated by the
    # bright zero-peak shoulder and the one-electron bump itself, pushing the
    # threshold so high that a genuine weak bump is missed. A low percentile tracks
    # the background between/around the peaks, so a real cluster clears it while
    # isolated single-bin spikes (rejected by the width requirement) do not.
    tail_mask = search_centers > (zero_peak_charge + 3 * zero_peak_width)
    tail_counts = smooth_search[tail_mask]
    tail_counts = tail_counts[tail_counts > 0]
    tail_background = float(np.percentile(tail_counts, 25)) if tail_counts.size else 1.0
    peak_prominence = max(2.0, 1.5 * np.sqrt(max(tail_background, 1.0)))
    peak_min_width = max(1.0, 0.5 * zero_peak_width / search_bin_width)
    peak_indices, _ = scipy_find_peaks(
        smooth_search,
        prominence=peak_prominence,
        distance=peak_distance,
        width=peak_min_width,
    )

    # Seed mu_1 on the one-electron bump, ranking candidates by their *excess above
    # the estimated zero-peak tail* rather than their raw counts. The bright but
    # monotonic zero-peak shoulder can carry more absolute counts than a real bump
    # sitting further out (especially with a small minimum gain, where the band's
    # lower edge is still in the tail), so ranking by raw counts seeds on the
    # shoulder; ranking by excess keeps it on the genuine bump. Prefer a detected
    # local maximum (find_peaks); fall back to the largest-excess bin when none is
    # found (a sparse or flat cluster find_peaks misses).
    zero_height = smooth_search[np.argmin(np.abs(search_centers - zero_peak_charge))]
    zero_tail_est = zero_height * np.exp(
        -(search_centers - zero_peak_charge) ** 2 / (2 * zero_peak_width ** 2))
    excess = smooth_search - zero_tail_est

    band_peaks = [pi for pi in peak_indices if one_lo <= search_centers[pi] <= one_hi]
    if band_peaks:
        found_one_peak = True
        seed_idx = max(band_peaks, key=lambda pi: excess[pi])
    else:
        found_one_peak = False
        band_idx = np.where((search_centers >= one_lo) & (search_centers <= one_hi))[0]
        seed_idx = band_idx[np.argmax(excess[band_idx])]
    one_peak_charge = search_centers[seed_idx]
    one_peak_height = smooth_search[seed_idx]

    # A user-supplied gain seed (a guess for the gain, ADU/e-) overrides the auto-
    # detected bump location: place the one-electron seed at zero_peak + gain_seed and
    # read its height off the histogram there. Everything downstream (the fit window
    # width, mu_1's seed/bounds, the amplitude guess) then follows from this guessed
    # gain. The post-fit acceptance band in calculate_noise_gain is unchanged, so a
    # seed outside [_MIN_GAIN_ADU, _MAX_GAIN_ADU] still only nudges the starting point.
    if gain_seed is not None:
        one_peak_charge = zero_peak_charge + float(gain_seed)
        seed_idx = int(np.argmin(np.abs(search_centers - one_peak_charge)))
        one_peak_height = smooth_search[seed_idx]

    gain_guess = one_peak_charge - zero_peak_charge  # within [_MIN_GAIN_ADU, _MAX_GAIN_ADU] when auto-detected

    left_halfwidth = max(4 * zero_peak_width, 0.5 * gain_guess)
    right_halfwidth = max(1.8 * gain_guess, 8 * zero_peak_width)
    zero_one_left = zero_peak_charge - window_left_scale * left_halfwidth
    zero_one_right = zero_peak_charge + window_right_scale * right_halfwidth
    zero_one_range = [zero_one_left, zero_one_right]

    # Scale the bin count with the (possibly widened) window so the bin width --
    # the ratio of bins to charge range -- stays the same as at scale 1.0.
    base_width = left_halfwidth + right_halfwidth
    scaled_width = window_left_scale * left_halfwidth + window_right_scale * right_halfwidth
    n_window = max(10, int(round(n * scaled_width / base_width))) if base_width > 0 else n

    _, zero_one_counts, zero_one_edges, zero_one_centers = _make_histogram(data, zero_one_range, n_window, max_bins=2000)
    max_zero_one_counts = max(zero_one_counts)

    if max_zero_one_counts == 0:
        raise ValueError(f'No data found in inferred zero-one range {zero_one_range}')

    fit_left, fit_right = zero_one_range
    # The pedestal is robustly located (it is the bulk of the data), so keep mu_0
    # on a short leash around the detected zero peak -- this too stops the zero
    # Gaussian from drifting into the one-electron region.
    m0_margin = max(0.5 * zero_peak_width, 2 * (zero_one_centers[1] - zero_one_centers[0]))
    # Give the fit headroom past both edges of the gain band so a real peak near an
    # edge is centred on its bump rather than pinned to it (a low-gain bump that
    # sits just above the floor would otherwise pin at the floor and read as the
    # minimum gain or be rejected). The lower headroom is floored at a few zero-peak
    # widths off the pedestal, so on a wide zero peak -- where the band's lower edge
    # is only a couple of sigma out -- the one-electron Gaussian still cannot slide
    # back onto the zero-peak shoulder. Acceptance is restricted to the band
    # [_MIN_GAIN_ADU, _MAX_GAIN_ADU] after the fit (in calculate_noise_gain).
    # Also keep mu_1 within ~one zero-peak width below the detected/seeded bump, so
    # the fit cannot slide down off the bump to mop up a heavy (non-Gaussian)
    # zero-peak tail -- the failure on a low-gain bump sitting on a steep tail.
    m1_low = max(one_lo - _GAIN_FIT_MARGIN, zero_peak_charge + 4 * zero_peak_width,
                 one_peak_charge - zero_peak_width)
    m1_high = min(fit_right, one_hi + _GAIN_FIT_MARGIN)
    if m1_high <= m1_low:
        m1_high = min(fit_right, one_hi)

    # Floor the shared width at a fraction of the zero-peak width: a real peak cannot
    # be narrower than the readout noise, so this stops curve_fit from collapsing the
    # Gaussians into a delta spike on a single noisy bin (the low-statistics failure
    # that otherwise yields a confident but bogus gain). popt layout is
    # (s, m0, m1, N0, N1): both peaks share the single width ``s``.
    sigma_floor = max(0.3 * zero_peak_width, (zero_one_centers[1] - zero_one_centers[0]) / 10, 1e-8)
    fit_bounds_low = [
        sigma_floor,
        max(fit_left, zero_peak_charge - m0_margin),
        m1_low,
        0,
        0,
    ]
    fit_bounds_high = [
        # The shared sigma is set by the well-determined pedestal width; bound it
        # tightly so the Gaussians cannot broaden to absorb the one-electron region
        # (which would drag mu_1 -- and the gain -- low on a weak cluster).
        2.0 * zero_peak_width,
        max(fit_right, zero_peak_charge + m0_margin),
        # mu_1's upper edge is the gain-band edge with headroom (m1_high, already
        # capped at fit_right). Do NOT scale it: when the pedestal is not subtracted
        # the zero peak sits at a (often negative) raw-ADU offset, so a multiplier
        # like 5*m1_high pushes the "upper" bound further from zero than m1_low and
        # inverts the bound (lower >= upper), which makes curve_fit raise. The post-
        # fit acceptance band [_MIN_GAIN_ADU, _MAX_GAIN_ADU] already caps the gain.
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
        one_peak_charge,
        max_zero_one_counts,
        one_peak_height,
    ]
    p0 = _clip_to_bounds(p0, fit_bounds)

    return zero_one_counts, zero_one_edges, p0, fit_bounds, zero_one_range, found_one_peak

#---------------- (1) Calculate noise / gain ----------------------------

def _one_electron_peak_is_real(centers, counts, popt, zero_peak_width):
    """Decide whether the fitted one-electron Gaussian is a genuine peak.

    The discriminating feature of a real one-electron peak -- as opposed to the
    fitter draping the second Gaussian over a heavy but *monotonic* tail of the
    zero peak -- is that, once the fitted zero-electron Gaussian is subtracted, the
    residual sits **higher in a band around mu_1 than in the valley between mu_1 and
    the zero peak**, and that band carries a statistically significant excess of
    pixels. A monotonic tail's residual only falls off with charge, so its "bump"
    never rises above the material just inside it and it is rejected -- this is the
    "is the second peak just inside the zero peak's envelope?" check.

    A strict local maximum is deliberately *not* required: a very low-statistics
    peak may be a few flat-topped bins of equal height, which `find_peaks` would
    miss. Comparing a band around mu_1 to the inner valley accepts such a plateau
    while still rejecting a featureless monotonic slope. The strength threshold is
    kept lenient so a weak-but-real cluster (a genuine low gain near 1 e-) passes.
    """
    s, m0, m1, N0, N1 = (popt[0], popt[1], popt[2], popt[3], popt[4])
    centers = np.asarray(centers, dtype=float)
    counts = np.asarray(counts, dtype=float)
    bin_width = centers[1] - centers[0]
    if bin_width <= 0:
        return False

    # Residual above the zero-electron Gaussian, smoothed on the zero-peak scale.
    zero_model = N0 * np.exp(-(centers - m0) ** 2 / (2 * s ** 2))
    resid = counts - zero_model
    smooth_win = max(3, int(round(zero_peak_width / bin_width)))
    resid_smooth = _smooth_counts(resid, window=smooth_win)

    # A band around mu_1 (the candidate peak). Use the median over the band so a
    # flat-topped low-statistics peak is treated the same as a sharp one.
    half = max(s, zero_peak_width)
    band = (centers >= m1 - half) & (centers <= m1 + half)
    if not np.any(band):
        return False
    band_level = float(np.median(resid_smooth[band]))

    # Compare the band to the residual on the shoulder immediately *inside* it
    # (toward the zero peak). A real peak turns up: the shoulder is the valley
    # between the zero and one peaks, so it sits below the band. A monotonic tail
    # of the zero peak only falls off, so its residual is still *higher* just inside
    # the band than in it -> rejected. Measuring the level right at the inner
    # shoulder (not the global minimum of a wide valley) is what catches a tail
    # that is still sloping down as it passes through the band.
    inner_lo = max(m0 + 2 * zero_peak_width, m1 - 2 * half)
    inner = (centers >= inner_lo) & (centers < m1 - half)
    inner_level = float(np.median(resid_smooth[inner])) if np.any(inner) else 0.0

    # The band must rise above that inner shoulder (rejects monotonic tails) ...
    if band_level <= inner_level:
        return False

    # ... and carry a significant integrated excess above the zero-peak envelope.
    total = counts[band].sum()
    background = zero_model[band].sum()
    excess = total - background
    if total <= 0:
        return False
    significance = excess / np.sqrt(total + 1.0)
    return (excess >= 4.0) and (significance >= 3.0)

def calculate_noise_gain(data, zero_one_test_range='auto', n=100, fit_bounds='default',
                         window_left_scale=1.0, window_right_scale=1.0,
                         peakfind_density=_PEAKFIND_DENSITY, gain_seed=None):

    data = np.array(data).flatten()
    data = data[np.isfinite(data)]

    if data.size == 0:
        raise ValueError('Input data contains no finite values')

    zero_one_counts, zero_one_edges, p0, auto_fit_bounds, zero_one_range, _found_one_peak = _auto_zero_one_setup(
        data,
        zero_one_test_range,
        n,
        window_left_scale=window_left_scale,
        window_right_scale=window_right_scale,
        peakfind_density=peakfind_density,
        gain_seed=gain_seed,
    )
    zero_one_centers = 0.5 * (zero_one_edges[:-1] + zero_one_edges[1:])

    if isinstance(fit_bounds, str) and fit_bounds in ('default', 'auto'):
        fit_bounds = auto_fit_bounds
    else:
        p0 = _clip_to_bounds(p0, fit_bounds)

    popt, pcov = curve_fit(double_gauss, zero_one_centers, zero_one_counts, p0=p0,
                           maxfev=20000, bounds=fit_bounds)

    # Extract pedestal, noise, and the double-Gaussian coefficients from the fit.
    s_fit, m0_fit, m1_fit, N0_fit, N1_fit = popt
    pedestal = m0_fit  # Pedestal is mean of the zero-electron peak
    noise = s_fit      # Noise is the shared peak width (std of the zero-electron peak)

    # Decide whether the one-electron Gaussian is a real peak or just the fitter
    # using the second Gaussian to absorb the (non-Gaussian) tail of the zero peak
    # (see _one_electron_peak_is_real). "No peak" therefore fires only when there
    # is genuinely no localized excess -- a real gain in the allowed band passes.
    # The gain (ADU/e-) is physically always > 1, so a fitted separation outside
    # [_MIN_GAIN_ADU, _MAX_GAIN_ADU] is rejected as not a real 0->1 step.
    # #_one_electron_peak_is_real(zero_one_centers, zero_one_counts, popt, noise) \
    gain_candidate = m1_fit - m0_fit
    if _MIN_GAIN_ADU <= gain_candidate <= _MAX_GAIN_ADU:
        gain = gain_candidate  # difference between the one- and zero-electron peak means
    else:
        # No trustworthy one-electron peak: gain is undefined. Return NaN and zero
        # the one-electron amplitude so downstream plotting draws only the single
        # (zero-electron) Gaussian.
        gain = np.nan
        popt = np.array([s_fit, m0_fit, m1_fit, N0_fit, 0.0])

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
            return 0.5, np.max(counts) * 1e4
        if yscale == 'linear':
            return 0, np.max(counts) + 2.5e4
        return None
    if ylim is None or ylim == 'none':
        if yscale != 'linear':
            return None  # let matplotlib pick a sensible (e.g. log) range per axis
        return 0, max(np.max(counts), np.max(fit_y)) * 1.05
    return ylim[0], ylim[1]

def _double_gauss_popt_electrons(double_gauss_popt, pedestal, gain):
    """Transform the converged ADU double-Gaussian fit into electron units.

    Converting charge to electrons, ``x_e = (x - pedestal) / gain``, is an exact
    linear rescaling under which the double Gaussian maps term for term: the means
    go to ``mu0_e = 0`` and ``mu1_e = 1`` (by construction, since ``pedestal == mu0``
    and ``gain == mu1 - mu0``), the widths scale as ``sigma / gain``, and the
    amplitudes are unchanged (the electron histogram uses the same bin count over
    the rescaled range, so every bin holds the same pixels). The electron-unit curve
    is therefore fully determined by the ADU fit.

    We transform analytically rather than re-fitting in electron units: an
    independent refit has nothing to gain (the answer is fixed by the transform) and
    its free parameters only let curve_fit drift the one-electron peak off
    ``mu_1 = 1`` and balloon the shared ``sigma`` -- exactly the spurious wide/displaced
    electron-space fit otherwise seen even when the ADU fit is clean.
    """
    s, m0, m1, N0, N1 = (double_gauss_popt[0], double_gauss_popt[1], double_gauss_popt[2],
                         double_gauss_popt[3], double_gauss_popt[4])
    return np.array([s / gain, (m0 - pedestal) / gain, (m1 - pedestal) / gain, N0, N1])

def _electron_double_gauss_popt(double_gauss_popt, pedestal, gain, centers_e, counts_e,
                                electron_fit_mode='transform'):
    """Double-Gaussian parameters in electron units, by one of two methods.

    ``'transform'`` (default): analytically rescale the converged ADU fit
    (``_double_gauss_popt_electrons`` -- exact, with ``mu_0 = 0`` / ``mu_1 = 1`` fixed
    by construction and no refit).

    ``'refit'``: fit the double Gaussian directly to the electron-unit histogram
    (``centers_e``/``counts_e``), starting from that analytic transform, so the
    electron-space peaks and widths are re-optimised rather than pinned by the ADU
    fit. Widths are kept positive and amplitudes non-negative; the means are left
    free, so the one-electron peak need not land exactly at 1 e-. Falls back to the
    analytic transform if the refit fails to converge. (See the note in
    ``_double_gauss_popt_electrons`` on why the transform is the default -- a refit
    can drift ``mu_1`` off 1 and balloon the shared ``sigma`` even on a clean ADU fit.)
    """
    transform = _double_gauss_popt_electrons(double_gauss_popt, pedestal, gain)
    if electron_fit_mode != 'refit':
        return transform
    lo = [1e-12, -np.inf, -np.inf, 0.0, 0.0]  # (s, m0, m1, N0, N1)
    hi = [np.inf] * 5
    try:
        popt_e, _ = curve_fit(double_gauss, centers_e, counts_e, p0=transform,
                              maxfev=20000, bounds=(lo, hi))
        return popt_e
    except (RuntimeError, ValueError):
        return transform

def _zero_one_label_adu(double_gauss_popt, gain):
    """Legend label for an ADU zero-one subplot.

    When `gain` is NaN no trustworthy one-electron peak was found, so only the
    zero-peak parameters are shown (the one-electron amplitude has been zeroed,
    making the plotted curve a single Gaussian) and the gain is reported as
    undefined rather than printing a meaningless value.
    """
    s, m0, m1, N0, N1 = (double_gauss_popt[0], double_gauss_popt[1], double_gauss_popt[2],
                         double_gauss_popt[3], double_gauss_popt[4])
    if np.isnan(gain):
        return (r'$\sigma$ = %5.3f, $\mu_0$ = %5.3f, $N_0$ = %5.3f' % (s, m0, N0)
                + '\n' + r'no $1\,e^{–}$ peak found (gain undefined)')
    return (r'$\sigma$ = %5.3f, $\mu_0$ = %5.3f, $\mu_1$ = %5.3f,' % (s, m0, m1)
            + '\n' + r'$N_0$ = %5.3f, $N_1$ = %5.3f, gain = %5.3f ADU/$e^{–}$' % (N0, N1, gain))

def plot_zero_one_peaks(data_ext,
                        zero_one_counts_ext,
                        zero_one_edges_ext,
                        pedestals, 
                        gains, 
                        double_gauss_popts, 
                        zero_one_ranges,
                        individual_figsize=_INDIVIDUAL_FIGSIZE,
                        subplots_figsize=_SUBPLOTS_FIGSIZE,
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
                        electron_fit_mode='transform',
                        plot_individual=False,
                        plot_together=True,
                        do_plot_adu=True,
                        sharex=False,
                        sharey=False,
                        show_titles=True,
                        save_plots=False,
                        show_plots=True,
                        fig_path='./', file='zero_one_peaks',
                        dpi=350):

    fig_path = Path(fig_path)
    # Build the output base from the source filename with its final extension stripped.
    # `base_name` is treated as a plain string and joined with the variant suffix and
    # '.jpeg' directly (see the save sites below) -- using Path.with_suffix/.with_stem
    # here would mistake any '.' in the stem (e.g. a voltage like VR-8.5) for the file
    # extension and truncate the name there.
    if file != 'zero_one_peaks':
        base_name = Path(file).stem + '_zero_one_peaks'
    else:
        base_name = file

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

            window_max = np.max(zero_one_counts)  # zero-peak height, drives y-scaling

            if yscale=='log':
                ax.set_yscale('log')
            elif yscale!='linear':
                ax.set_yscale(yscale)
            ax.stairs(_bar_heights(zero_one_counts, yscale), zero_one_edges, fill=True)

            curve_x = _fit_curve_x(zero_one_range[0], zero_one_range[1], len(zero_one_counts))
            ax.plot(curve_x, double_gauss(curve_x, *double_gauss_popt), 'r',
                label=_zero_one_label_adu(double_gauss_popt, gain))
            ax.legend(loc="upper right", fontsize=fontsize)

            if xlim=='default':
                ax.set_xlim(zero_one_range[0], zero_one_range[1])
            elif xlim is not None and xlim!='none':
                ax.set_xlim(xlim)
            # xlim is None or 'none': leave autoscaled to the full data extent

            if ylim=='default':
                if yscale=='log':
                    ax.set_ylim(0.5, window_max * 1e4)
                elif yscale=='linear':
                    ax.set_ylim(0, window_max + 3e4)
            elif ylim!='none':
                ax.set_ylim(ylim)

            if save_plots:
                output_path = fig_path / f'{base_name}_EXT{ext+1}.jpeg'
                plt.savefig(str(output_path), dpi=dpi)
                print(f'Saved plots to {output_path}')
            _finish_fig(show_plots)

        if do_convert_to_electrons:
            for ext, data in enumerate(data_ext):
                data=np.array(data).flatten()
                zero_one_range = zero_one_ranges[ext]
                pedestal = pedestals[ext]
                gain = gains[ext]

                # No one-electron peak -> no gain -> electron conversion (divide by
                # gain) is undefined; skip the electron plot for this extension.
                if np.isnan(gain):
                    continue

                data_window=data[(data > zero_one_range[0]) & (data < zero_one_range[1])]

                data_window_e = convert_to_electrons(data_window, pedestal, gain)
                zero_one_range_e = convert_to_electrons(zero_one_range, pedestal, gain)
                # Match the (scale-adjusted) ADU fit-window bin count so the electron
                # plot keeps the same bins-to-window ratio when the window is widened.
                nbins = len(zero_one_counts_ext[ext])

                # Window histogram (drawn as the bars).
                zero_one_counts_e, zero_one_edges_e = np.histogram(data_window_e, bins=nbins, range=zero_one_range_e)
                zero_one_centers_e = 0.5 * (zero_one_edges_e[:-1] + zero_one_edges_e[1:])
                window_max_e = np.max(zero_one_counts_e)

                # Electron-unit curve: the ADU fit rescaled exactly, or a direct refit in
                # electron units (see electron_fit_mode / _electron_double_gauss_popt).
                double_gauss_popt_e = _electron_double_gauss_popt(
                    double_gauss_popts[ext], pedestal, gain,
                    zero_one_centers_e, zero_one_counts_e, electron_fit_mode)

                fig, ax = plt.subplots(1, 1, figsize=individual_figsize, constrained_layout=True)
                if show_titles:
                    ax.set_title(rf'{additional_title}{suptitle} (Nimages = {nimages}): EXT {ext + 1}', fontsize=12, pad=10)

                if yscale=='log':
                    ax.set_yscale('log')
                elif yscale!='linear':
                    ax.set_yscale(yscale)

                ax.stairs(_bar_heights(zero_one_counts_e, yscale), zero_one_edges_e, fill=True)
                ax.set_xlabel(r'Charge ($e^–$)')
                ax.set_ylabel('N')
                curve_x_e = _fit_curve_x(zero_one_range_e[0], zero_one_range_e[1], len(zero_one_counts_e))
                ax.plot(curve_x_e, double_gauss(curve_x_e, *double_gauss_popt_e), 'r',
                    label=r'$\sigma$ = %5.3f $e^{–}$, $\mu_0$ = %5.3f $e^{–}$, $\mu_1$ = %5.3f $e^{–}$,'%tuple(double_gauss_popt_e)[0:3]
                    +'\n'+'gain = %5.3f ADU/$e^{–}$'%gain)
                ax.legend(loc="upper right", fontsize=fontsize)

                if xlim=='default':
                    ax.set_xlim(zero_one_range_e[0], zero_one_range_e[1])
                elif xlim is not None and xlim!='none':
                    ax.set_xlim(xlim)
                # xlim is None or 'none': leave autoscaled to the full data extent

                if ylim_electrons=='default':
                    if yscale=='log':
                        ax.set_ylim(0.5, window_max_e * 1e4)
                    elif yscale=='linear':
                        ax.set_ylim(0, window_max_e + 2.5e4)
                elif ylim_electrons!='none':
                    ax.set_ylim(ylim_electrons)

                if save_plots:
                    output_path = fig_path / f'{base_name}_electrons_EXT{ext+1}.jpeg'
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

                if yscale=='log':
                    ax.set_yscale('log')
                elif yscale!='linear':
                    ax.set_yscale(yscale)

                ax.stairs(_bar_heights(zero_one_counts, yscale), zero_one_edges, fill=True)

                curve_x = _fit_curve_x(zero_one_range[0], zero_one_range[1], len(zero_one_counts))
                fit_y = double_gauss(curve_x, *double_gauss_popt)
                ax.plot(curve_x, fit_y, 'r',
                    label=_zero_one_label_adu(double_gauss_popt, gain))

                ax.set_xlabel('Charge (ADU)')
                ax.set_ylabel('N')
                ax.set_title(f'EXT {ext + 1}')
                ax.legend(loc="upper right", fontsize=fontsize - 2)

                if xlim=='default':
                    ax.set_xlim(zero_one_range[0], zero_one_range[1])
                elif xlim is not None and xlim!='none':
                    ax.set_xlim(xlim)
                # xlim is None or 'none': leave autoscaled to the full data extent

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
                output_path = fig_path / f'{base_name}.jpeg'
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

                # No one-electron peak -> gain undefined -> cannot convert to
                # electrons; leave the subplot empty with a note and move on.
                if np.isnan(gain):
                    ax.set_title(f'EXT {ext + 1}')
                    ax.set_xlabel(r'Charge ($e^–$)')
                    ax.set_ylabel('N')
                    ax.text(0.5, 0.5, r'no $1\,e^{–}$ peak found',
                            ha='center', va='center', transform=ax.transAxes)
                    _ylims.append(None)
                    continue

                data_window = data[(data > zero_one_range[0]) & (data < zero_one_range[1])]

                data_window_e = convert_to_electrons(data_window, pedestal, gain)
                zero_one_range_e = convert_to_electrons(zero_one_range, pedestal, gain)
                # Match the (scale-adjusted) ADU fit-window bin count so the electron
                # plot keeps the same bins-to-window ratio when the window is widened.
                nbins = len(zero_one_counts_ext[ext])

                # Window histogram (drawn as the bars).
                zero_one_counts_e, zero_one_edges_e = np.histogram(data_window_e, bins=nbins, range=zero_one_range_e)
                zero_one_centers_e = 0.5 * (zero_one_edges_e[:-1] + zero_one_edges_e[1:])

                # Electron-unit curve: the ADU fit rescaled exactly, or a direct refit in
                # electron units (see electron_fit_mode / _electron_double_gauss_popt).
                double_gauss_popt_e = _electron_double_gauss_popt(
                    double_gauss_popts[ext], pedestal, gain,
                    zero_one_centers_e, zero_one_counts_e, electron_fit_mode)

                if yscale=='log':
                    ax.set_yscale('log')
                elif yscale!='linear':
                    ax.set_yscale(yscale)

                ax.stairs(_bar_heights(zero_one_counts_e, yscale), zero_one_edges_e, fill=True)

                ax.set_title(f'EXT {ext + 1}')
                curve_x_e = _fit_curve_x(zero_one_range_e[0], zero_one_range_e[1], len(zero_one_counts_e))
                fit_y_e = double_gauss(curve_x_e, *double_gauss_popt_e)
                ax.plot(curve_x_e, fit_y_e, 'r',
                    label=r'$\sigma$ = %5.3f $e^{–}$, $\mu_0$ = %5.3f $e^{–}$, $\mu_1$ = %5.3f $e^{–}$,'%tuple(double_gauss_popt_e)[0:3]
                    +'\n'+'gain = %5.3f ADU/$e^{–}$'%gain)
                ax.legend(loc="upper right", fontsize=fontsize - 2)
                ax.set_xlabel(r'Charge ($e^–$)')
                ax.set_ylabel('N')

                if xlim=='default':
                    ax.set_xlim(zero_one_range_e[0], zero_one_range_e[1])
                elif xlim is not None and xlim!='none':
                    ax.set_xlim(xlim)
                # xlim is None or 'none': leave autoscaled to the full data extent

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
                output_path = fig_path / f'{base_name}_electrons.jpeg'
                plt.savefig(str(output_path), dpi=dpi)
                print(f'Saved plot to {output_path}')
            _finish_fig(show_plots)

#---------------- Plotting: dark-current window ----------------------------

def _dark_current_fit_label(sigma_e, gain, N1):
    """Legend text for the red fit curve: shared width (e-), gain, and 1 e- amplitude N1."""
    return (r'$\sigma$ = %.3f $e^-$' % sigma_e
            + '\n' + r'gain = %.3f ADU/$e^-$, $N_1$ = %.4g' % (gain, N1))

def plot_dark_current_zero_one(data_ext, zero_one_counts_ext, zero_one_edges_ext,
                               pedestals, gains, double_gauss_popts, zero_one_ranges,
                               dark_current_rows, count_center='one_electron',
                               count_nsigma=1.0, electron_fit_mode='transform',
                               nimages=10, figsize=_SUBPLOTS_FIGSIZE, fontsize=10, yscale='log',
                               suptitle='Dark-Current Window (Zero-One Peaks, electrons)',
                               additional_title='', show_titles=True,
                               save_plots=False, show_plots=True,
                               fig_path='./', file='dark_current', dpi=350):
    """Plot the electron-unit zero/one distributions with the dark-current count window.

    One subplot per extension (2x2): the electron-unit histogram and fitted
    double-Gaussian curve, and a legend with the shared sigma, the gain and the dark
    current(s). When the 'count' method was used, vertical dashed lines mark the
    +/- ``count_nsigma`` * sigma charge window it integrates around 1 e-; otherwise
    those lines are omitted and (when the 'weighted' method was used) the legend
    instead reports that method's dark current and its formula. Extensions with no
    one-electron peak (gain undefined) cannot be converted to electrons and are left
    with an explanatory note.
    """
    fig_path = Path(fig_path)
    base_name = Path(file).stem + '_dark_current' if file != 'dark_current' else file

    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    if show_titles:
        fig.suptitle(f'{additional_title}{suptitle} (Nimages = {nimages})')
    axs = axs.flatten()

    for ext, data in enumerate(data_ext):
        ax = axs[ext]
        ax.set_title(f'EXT {ext + 1}')
        ax.set_xlabel(r'Charge ($e^-$)')
        ax.set_ylabel('N')

        pedestal = pedestals[ext]
        gain = gains[ext]
        if np.isnan(gain):
            ax.text(0.5, 0.5, r'no $1\,e^-$ peak found', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        data = np.asarray(data).flatten()
        zero_one_range = zero_one_ranges[ext]
        data_window = data[(data > zero_one_range[0]) & (data < zero_one_range[1])]
        data_window_e = convert_to_electrons(data_window, pedestal, gain)
        zero_one_range_e = convert_to_electrons(np.asarray(zero_one_range), pedestal, gain,
                                                flatten=False)
        nbins = len(zero_one_counts_ext[ext])
        counts_e, edges_e = np.histogram(data_window_e, bins=nbins, range=zero_one_range_e)
        centers_e = 0.5 * (edges_e[:-1] + edges_e[1:])

        popt_e = _electron_double_gauss_popt(double_gauss_popts[ext], pedestal, gain,
                                             centers_e, counts_e, electron_fit_mode)
        sigma_e, N1 = popt_e[0], popt_e[4]

        if yscale != 'linear':
            ax.set_yscale(yscale)
        ax.stairs(_bar_heights(counts_e, yscale), edges_e, fill=True, color='#4e117d')

        curve_x_e = _fit_curve_x(zero_one_range_e[0], zero_one_range_e[1], len(counts_e))
        ax.plot(curve_x_e, double_gauss(curve_x_e, *popt_e), '#8dde1b',
                label=_dark_current_fit_label(sigma_e, gain, N1))

        dc_count = dark_current_rows[ext].get('dark_current_count_e_per_pix_day')
        if dc_count is not None:
            # 'count' method was used: draw its +/- count_nsigma * sigma window about the
            # 1 e- charge, computed in ADU exactly as the method does, then converted to
            # electrons so the lines match the pixels actually counted (independent of
            # any refit). Label it with the method's dark current.
            popt = double_gauss_popts[ext]
            sigma_adu = popt[0]
            center_adu = popt[2] if count_center == 'mu1' else pedestal + gain
            lo_e = (center_adu - count_nsigma * sigma_adu - pedestal) / gain
            hi_e = (center_adu + count_nsigma * sigma_adu - pedestal) / gain
            window_label = (rf'count window ($\pm${count_nsigma:g}$\,\sigma$)'
                            + '\n' + r'dark current = %.3g $e^-$/pix/day' % dc_count)
            ax.axvline(lo_e, color='k', linestyle=':', linewidth=1.2, label=window_label)
            ax.axvline(hi_e, color='k', linestyle=':', linewidth=1.2)
        else:
            # 'count' not used: no window lines. If the 'weighted' method was used,
            # report its dark current and formula as a legend-only entry.
            dc_weighted = dark_current_rows[ext].get('dark_current_weighted_e_per_pix_day')
            if dc_weighted is not None:
                ax.plot([], [], ' ',
                        label=(r'dark current = %.3g $e^-$/pix/day' % dc_weighted
                               + '\n' + r'$n_{events}$ / img $ = N_1$ / $(N_1 + N_0)$'))

        ax.set_xlim(zero_one_range_e[0], zero_one_range_e[1])
        ax.legend(loc='upper right', fontsize=fontsize - 2)

    if save_plots:
        output_path = fig_path / f'{base_name}.jpeg'
        plt.savefig(str(output_path), dpi=dpi)
        print(f'Saved plot to {output_path}')
    _finish_fig(show_plots)

def plot_charge_per_column(data_ext, n_std_to_mask=1.5, fit_cols_ext=None,
                           figsize=_SUBPLOTS_FIGSIZE, additional_title='', show_titles=True,
                           nimages=1, verbose=False, save_plots=False, show_plots=True,
                           fig_path='./', file='charge_per_column', dpi=350):
    """Median charge per column for each extension, as a 2x2 grid of scatter plots.

    Intended for the *raw* (pre-pedestal-subtraction) data, so the per-column medians
    reveal which columns carry anomalous charge. A column is flagged "hot" (drawn red)
    when its median sits at least ``n_std_to_mask`` biweight standard deviations from
    the biweight location of the extension -- the same robust threshold the pedestal
    subtraction uses to mask outliers. Every column is plotted; ``fit_cols_ext``, if
    given, is a per-extension ``(c0, c1)`` slice (or None) marking which columns the
    zero/one fit keeps -- the columns *outside* it are shaded light grey, so the plot
    shows the full frame while making the masked-out region visible.
    """
    fig_path = Path(fig_path)
    base_name = Path(file).stem + '_charge_per_column' if file != 'charge_per_column' else file

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    if show_titles:
        fig.suptitle(f'{additional_title}Median Charge per Column (Nimages = {nimages})')

    for ext, data in enumerate(data_ext):
        ax = axes.flat[ext]
        data = np.asarray(data)
        col_idx = np.arange(data.shape[1])
        median_charge = np.median(data, axis=0)

        flat = data.flatten()
        loc = biweight_location(flat, ignore_nan=True)
        scale = np.sqrt(biweight_midvariance(flat, ignore_nan=True))
        hot = np.abs(median_charge - loc) >= n_std_to_mask * scale

        ax.scatter(col_idx[~hot], median_charge[~hot], s=1)
        ax.scatter(col_idx[hot], median_charge[hot], s=1, color='red')

        # Shade the columns excluded by fit_cols (Python-slice semantics handle negative
        # or None endpoints). Each maximal run of excluded columns is one grey span, so a
        # left and/or right cut both show; nothing is drawn when all columns are kept.
        fc = fit_cols_ext[ext] if fit_cols_ext is not None else None
        if fc is not None:
            kept = np.zeros(data.shape[1], dtype=bool)
            kept[fc[0]:fc[1]] = True
            edges = np.diff(np.concatenate(([0], (~kept).astype(np.int8), [0])))
            for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                ax.axvspan(start - 0.5, end - 0.5, color='gray', alpha=0.15, linewidth=0)

        if show_titles:
            ax.set_title(f'EXT {ext + 1}')
        if ext % 2 == 0:        # left column of the grid
            ax.set_ylabel('Median Charge (ADU)')
        if ext // 2 == 1:       # bottom row of the grid
            ax.set_xlabel('Column')

        if verbose:
            pct = 100 * hot.sum() / len(hot) if len(hot) else 0.0
            print(f'EXT {ext + 1}: {int(hot.sum())} hot columns ({pct:.1f}%) '
                  f'(median charge >= {n_std_to_mask} SDs from pedestal location): '
                  f'{col_idx[hot].tolist()}')

    if save_plots:
        output_path = fig_path / f'{base_name}.jpeg'
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

def get_exposure_time_days(file_input):
    """Exposure time in days, read from the FITS headers as ``DATEEND - DATEINI``.

    Scans every HDU (primary + extensions) for the ISO-8601 ``DATEINI`` (start) and
    ``DATEEND`` (end) timestamps and returns their difference converted to days.

    The exposure time is the per-image integration time even for a stitched frame:
    each pixel of a stitched image was exposed for the same duration as a single
    image (stitching adds pixels, not exposure), and ``stitch_fits`` copies the
    first image's headers, so this returns that single image's exposure as required.

    Returns ``np.nan`` if either timestamp is missing.
    """
    file_path = Path(file_input).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"FITS file not found: {file_path}")

    dateini = dateend = None
    with fits.open(str(file_path)) as hdu_list:
        for hdu in hdu_list:
            if dateini is None and 'DATEINI' in hdu.header:
                dateini = hdu.header['DATEINI']
            if dateend is None and 'DATEEND' in hdu.header:
                dateend = hdu.header['DATEEND']
            if dateini is not None and dateend is not None:
                break

    if dateini is None or dateend is None:
        return np.nan

    t_start = datetime.fromisoformat(str(dateini))
    t_end = datetime.fromisoformat(str(dateend))
    return (t_end - t_start).total_seconds() / 86400.0

def get_pixel_binning(file_input):
    """(NPBIN, NSBIN) = vertical/horizontal pixel binning, from the FITS headers.

    The detector is read out binned, so each image pixel holds the summed charge of
    NPBIN (vertical) x NSBIN (horizontal) physical CCD pixels. Scans every HDU for the
    keys and returns them as ints; a missing key is returned as ``None`` so the caller
    can warn and fall back to 1 (no binning).
    """
    file_path = Path(file_input).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"FITS file not found: {file_path}")

    npbin = nsbin = None
    with fits.open(str(file_path)) as hdu_list:
        for hdu in hdu_list:
            if npbin is None and 'NPBIN' in hdu.header:
                npbin = int(hdu.header['NPBIN'])
            if nsbin is None and 'NSBIN' in hdu.header:
                nsbin = int(hdu.header['NSBIN'])
            if npbin is not None and nsbin is not None:
                break
    return npbin, nsbin

def raw_pedestal_locations(data_ext, use_biweight=False):
    """Robust per-extension pedestal (zero-peak) location of the RAW data.

    Intended to be called on the raw extensions *before* any pedestal subtraction,
    so the reported baseline is the physical pedestal level rather than the ~0
    baseline left after row/column subtraction. Returns one value per extension in
    ADU: the median of the (finite) pixels by default -- robust to the sparse
    positive-charge tail since the zero-electron peak dominates -- or the Tukey
    biweight location when ``use_biweight=True``.
    """
    locations = []
    for data in data_ext:
        flat = np.asarray(data, dtype=float).flatten()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            locations.append(np.nan)
        elif use_biweight:
            locations.append(float(biweight_location(flat)))
        else:
            locations.append(float(np.median(flat)))
    return locations

# CCD-geometry header keys used to locate the serial overscan columns.
_OVERSCAN_HEADER_KEYS = ('PRESCAN', 'PHYSCOL', 'NCOL', 'NCOLPRE', 'NSBIN')

def get_fits_header(file_input):
    """Return the first FITS header (primary or extension) that carries the CCD
    geometry keys in `_OVERSCAN_HEADER_KEYS`, or the primary header if none do."""
    file_path = Path(file_input).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"FITS file not found: {file_path}")
    with fits.open(str(file_path)) as hdu_list:
        for hdu in hdu_list:
            if all(k in hdu.header for k in _OVERSCAN_HEADER_KEYS):
                return hdu.header
        return hdu_list[0].header

def overscan_cols_from_header(header):
    """Compute the serial-overscan column slice ``(c0, c1)`` from CCD geometry keys.

    The CCD has PRESCAN inactive columns, then PHYSCOL physical columns, then the
    overscan. A frame reads CCD columns ``[NCOLPRE*NSBIN, (NCOL+NCOLPRE)*NSBIN]``,
    so the overscan is every CCD column past ``PRESCAN + PHYSCOL``. Dividing that
    CCD-column count by NSBIN converts it to *binned image* columns, giving the
    last ``n = ((NCOL+NCOLPRE)*NSBIN - (PRESCAN+PHYSCOL)) // NSBIN`` columns of the
    image (for NSBIN=1 this is just the CCD-column count).

    Returns ``(-n, None)`` (a half-open slice selecting the last ``n`` columns),
    or ``None`` if a required key is missing or the computed count is not positive.
    """
    if header is None or not all(k in header for k in _OVERSCAN_HEADER_KEYS):
        return None
    prescan = int(header['PRESCAN'])
    physcol = int(header['PHYSCOL'])
    ncol = int(header['NCOL'])
    ncolpre = int(header['NCOLPRE'])
    nsbin = int(header['NSBIN'])
    if nsbin <= 0:
        return None
    n_overscan_ccd = (ncol + ncolpre) * nsbin - (prescan + physcol)
    n_overscan = n_overscan_ccd // nsbin  # CCD columns -> binned image columns
    if n_overscan <= 0:
        return None
    return (-n_overscan, None)

#---------------- Per-extension zero/one fitting ----------------------------

def _value_for_extension(value, ext, n_ext):
    if isinstance(value, (list, tuple)) and len(value) == n_ext:
        if all(isinstance(v, (list, tuple, np.ndarray)) and len(v) == 2 for v in value):
            return value[ext]
    return value

def _scalar_for_extension(value, ext, n_ext):
    """Select a per-extension scalar from ``value``.

    ``value`` may be a single number/None applied to every extension, or a
    length-``n_ext`` list/tuple of per-extension entries (each a number or None).
    Anything else (e.g. a scalar) is returned unchanged for every extension.
    """
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == n_ext:
        return value[ext]
    return value

def get_zero_one_peaks_ext(data_ext,
                           n=100, fit_bounds='default', zero_one_test_range='auto',
                           window_left_scale=1.0, window_right_scale=1.0,
                           peakfind_density=_PEAKFIND_DENSITY, gain_seed=None):
    zero_one_counts_ext = []
    zero_one_edges_ext = []
    pedestals = []
    gains = []
    double_gauss_popts = []
    zero_one_ranges = []
    for ext, data in enumerate(data_ext):

        zero_one_test_range_ext = _value_for_extension(zero_one_test_range, ext, len(data_ext))
        gain_seed_ext = _scalar_for_extension(gain_seed, ext, len(data_ext))

        zero_one_counts, zero_one_edges, pedestal, noise, gain, double_gauss_popt, zero_one_range = calculate_noise_gain(
            data,
            zero_one_test_range=zero_one_test_range_ext,
            n=n,
            fit_bounds=fit_bounds,
            window_left_scale=window_left_scale,
            window_right_scale=window_right_scale,
            peakfind_density=peakfind_density,
            gain_seed=gain_seed_ext,
        )
        zero_one_counts_ext.append(zero_one_counts)
        zero_one_edges_ext.append(zero_one_edges)
        pedestals.append(pedestal)
        gains.append(gain)
        double_gauss_popts.append(double_gauss_popt)
        zero_one_ranges.append(zero_one_range)

    return zero_one_counts_ext, zero_one_edges_ext, pedestals, gains, double_gauss_popts, zero_one_ranges


#---------------- Dark current ----------------------------

# CSV column name for each dark-current counting method. Only the chosen method(s)
# are computed and written, so the user can run one or several and compare them.
_DARK_CURRENT_COLUMN = {
    'count': 'dark_current_count_e_per_pix_day',
    'integrate': 'dark_current_integrate_e_per_pix_day',
    'weighted': 'dark_current_weighted_e_per_pix_day',
}

# Ordered list of every method, used both to expand 'all' and to order CSV columns.
_DARK_CURRENT_METHODS = ('count', 'integrate', 'weighted')

def _resolve_dark_current_methods(method):
    """Methods to compute, from a 'count'/'integrate'/'weighted'/'all' selector."""
    if method == 'all':
        return _DARK_CURRENT_METHODS
    if method in _DARK_CURRENT_METHODS:
        return (method,)
    raise ValueError("dark current method must be one of 'count', 'integrate', "
                     f"'weighted', or 'all'; got {method!r}")

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
    return count / erf(nsigma / np.sqrt(2))  # scale window count up to the full Gaussian

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
    return area / bin_width             # -> number of pixels

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

    ``method`` may be ``'count'``, ``'integrate'``, ``'weighted'`` or ``'all'`` -- only
    the requested method(s) are computed.

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

#---------------- Per-extension summary (CSV) ----------------------------

# Column order of the per-extension summary. Extends the nonlinearity_studies columns
# (ext, gain_adu_per_e, noise_e) with noise_adu, the zero-peak width in raw ADU.
_EXTENSION_SUMMARY_FIELDS = ['ext', 'gain_adu_per_e', 'noise_adu', 'noise_e']

def build_extension_summary(gains, double_gauss_popts, dark_current_rows=None,
                            exposure_days=None, raw_pedestals=None):
    """Per-extension rows of gain (ADU/e-) and noise, in both ADU and e-.

    ``noise_adu`` is the zero-electron peak width (``s0``, the first double-Gaussian
    coefficient, in ADU); ``noise_e`` is that same width divided by the gain. When no
    one-electron peak was found the gain is NaN, so noise_e is NaN as well, but
    noise_adu is still reported (it does not depend on the gain).

    When supplied, optional columns are merged in: ``raw_pedestals`` adds
    ``pedestal_raw_adu`` (the pre-subtraction baseline, see ``raw_pedestal_locations``),
    ``exposure_days`` adds an ``exposure_days`` column, and ``dark_current_rows`` (from
    ``calculate_dark_current``) adds the chosen dark-current column(s).
    """
    rows = []
    for ext, gain in enumerate(gains):
        noise_adu = double_gauss_popts[ext][0]  # s0: std of the zero-electron peak in ADU
        row = {'ext': ext + 1}
        if raw_pedestals is not None:
            row['pedestal_raw_adu'] = raw_pedestals[ext]
        row['gain_adu_per_e'] = gain
        row['noise_adu'] = noise_adu
        row['noise_e'] = noise_adu / gain  # ADU -> e- by dividing by gain (ADU/e-)
        if exposure_days is not None:
            row['exposure_days'] = exposure_days
        if dark_current_rows is not None:
            row.update(dark_current_rows[ext])
        rows.append(row)
    return rows

def _extension_summary_fieldnames(dark_current_rows=None, exposure_days=None,
                                  raw_pedestals=None):
    """Ordered CSV field names for the per-extension summary, including the optional
    raw-pedestal, exposure-time and dark-current columns actually present."""
    fieldnames = ['ext']
    if raw_pedestals is not None:
        fieldnames.append('pedestal_raw_adu')
    fieldnames += ['gain_adu_per_e', 'noise_adu', 'noise_e']
    if exposure_days is not None:
        fieldnames.append('exposure_days')
    if dark_current_rows:
        for method in _DARK_CURRENT_METHODS:
            column = _DARK_CURRENT_COLUMN[method]
            if any(column in row for row in dark_current_rows):
                fieldnames.append(column)
    return fieldnames

def write_extension_summary_csv(save_path, gains, double_gauss_popts,
                                dark_current_rows=None, exposure_days=None,
                                raw_pedestals=None):
    """Write the per-extension gain/noise (and optional dark-current) summary as CSV.

    Returns the list of summary rows (see build_extension_summary).
    """
    rows = build_extension_summary(gains, double_gauss_popts, dark_current_rows,
                                   exposure_days, raw_pedestals)
    fieldnames = _extension_summary_fieldnames(dark_current_rows, exposure_days,
                                               raw_pedestals)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return rows
