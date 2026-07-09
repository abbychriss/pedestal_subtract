"""
fit_zero_one — split out of core.py.
"""
import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks

from .calibrate import calculate_noise_gain
from .constants import (
    _MAX_GAIN_ADU,
    _MIN_GAIN_ADU,
    _ZERO_ONE_N_BINS,
    _GAIN_SEED_FIT_MARGIN,
)
from .fits_io import _scalar_for_extension, _value_for_extension

_GAIN_FIT_MARGIN = 0.3


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


def _clamp_n_bins(n_bins, min_bins=10, max_bins=4000):
    """Clamp the user-supplied strict bin count into a sane range.

    This is the single binning rule used everywhere in the zero/one fitter: every
    window (test range, peak-search range, fit window) is divided into ``n_bins`` bins,
    so the bin *width* scales with how wide the window is.
    """
    return min(max_bins, max(min_bins, int(round(n_bins))))


def _make_histogram(data, hist_range, n, max_bins=4000):
    left, right = hist_range
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        raise ValueError(f'Invalid histogram range: {hist_range}')

    # `n` is the desired number of bins spanning this range (already clamped by the
    # caller), so the histogram gets ~n bins over this range.
    nbins = min(max_bins, max(10, int(n)))
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


def _auto_zero_one_setup(data, zero_one_test_range, n_bins=_ZERO_ONE_N_BINS,
                         window_left_scale=1.0, window_right_scale=1.0, gain_seed=None):
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
    data_test, counts_test, edges_test, centers_test = _make_histogram(
        data, test_range, _clamp_n_bins(n_bins))

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
    # The half-max walk overestimates the width on a coarse histogram (it can only
    # step in whole bins), and everything downstream -- the mu_1 lower bound, the
    # smoothing scale, the sigma bounds -- inflates with it. Cap it with the robust
    # MAD sigma of the windowed data, which is binning-independent.
    test_median = float(np.median(data_test))
    robust_test_sigma = 1.4826 * float(np.median(np.abs(data_test - test_median)))
    if robust_test_sigma > 0:
        zero_peak_width = min(zero_peak_width, robust_test_sigma)
    zero_peak_width = max(zero_peak_width, bin_width)

    one_lo = zero_peak_charge + _MIN_GAIN_ADU
    one_hi = zero_peak_charge + _MAX_GAIN_ADU
    search_left = zero_peak_charge - 5 * zero_peak_width
    search_right = one_hi + 3 * zero_peak_width

    search_range = (search_left, search_right)
    _, search_counts, search_edges, search_centers = _make_histogram(
        data, search_range, _clamp_n_bins(n_bins))
    search_bin_width = search_centers[1] - search_centers[0]
    # Smooth on the scale of the zero-peak width rather than re-binning coarsely.
    # In a low-statistics image the single-electron hits are sparse 1-count bins,
    # so on the raw fine histogram they never form a bump; averaging over ~one
    # zero-peak width lets a real cluster aggregate into a detectable peak while a
    # smooth monotonic tail stays bump-free. With coarse bins (low n_bins)
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
        # No detected local maximum: seed on the largest excess in the band, but only
        # past the zero-peak shoulder (zero + 4 widths). The Gaussian tail estimate is
        # only approximate, so just above one_lo the "excess" is dominated by the
        # mis-modelled bright shoulder, not by one-electron events -- seeding there
        # collapses the fit onto the shoulder. If the shoulder spans the whole band
        # (very wide zero peak), fall back to the full band.
        seed_lo = max(one_lo, zero_peak_charge + 4 * zero_peak_width)
        band_idx = np.where((search_centers >= seed_lo) & (search_centers <= one_hi))[0]
        if band_idx.size == 0:
            band_idx = np.where((search_centers >= one_lo) & (search_centers <= one_hi))[0]
        if band_idx.size == 0:
            # A wide zero peak plus a strict (range-independent) bin count can leave the
            # narrow [_MIN_GAIN_ADU, _MAX_GAIN_ADU] band without a single search bin
            # center in it. Seed on the search bin nearest the band midpoint instead of
            # crashing -- the fit window/bounds below still constrain mu_1 to the band.
            band_mid = 0.5 * (one_lo + one_hi)
            seed_idx = int(np.argmin(np.abs(search_centers - band_mid)))
        else:
            seed_idx = band_idx[np.argmax(excess[band_idx])]
    one_peak_charge = search_centers[seed_idx]

    # The auto-detected bump location (within [_MIN_GAIN_ADU, _MAX_GAIN_ADU] when a
    # bump was found) sizes the fit window below. Keep it separate from any user gain
    # seed so the guess can never change the window width or the binning.
    auto_one_peak_charge = one_peak_charge
    auto_gain_guess = auto_one_peak_charge - zero_peak_charge

    # A user-supplied gain seed (a guess for the gain, ADU/e-) repositions ONLY the
    # starting point/bound for mu_1: place the one-electron seed at zero_peak +
    # gain_seed (N_1's seed/bounds are read off the fit histogram there, below). It
    # deliberately does not feed the window half-widths or the bin density, so the
    # fit window and histogram resolution are identical with or without a guess.
    # mu_1's fit bounds are pinned to a tight band around this seed below; the
    # post-fit acceptance band in calculate_noise_gain is unchanged.
    if gain_seed is not None:
        one_peak_charge = zero_peak_charge + float(gain_seed)

    # Fit window: sized from the auto-detected gain only (never the seed).
    left_halfwidth = max(4 * zero_peak_width, 0.5 * auto_gain_guess)
    right_halfwidth = max(1.8 * auto_gain_guess, 8 * zero_peak_width)
    zero_one_left = zero_peak_charge - window_left_scale * left_halfwidth
    zero_one_right = zero_peak_charge + window_right_scale * right_halfwidth
    zero_one_range = [zero_one_left, zero_one_right]

    # Binning is a strict bin count: n_bins bins span the window regardless of its
    # width, so widening the window (via the scales) coarsens the bin width instead of
    # adding bins.
    n_window = _clamp_n_bins(n_bins)

    _, zero_one_counts, zero_one_edges, zero_one_centers = _make_histogram(data, zero_one_range, n_window, max_bins=4000)
    max_zero_one_counts = max(zero_one_counts)

    if max_zero_one_counts == 0:
        raise ValueError(f'No data found in inferred zero-one range {zero_one_range}')

    fit_left, fit_right = zero_one_range
    # The pedestal is robustly located (it is the bulk of the data), so keep mu_0
    # on a short leash around the detected zero peak -- this too stops the zero
    # Gaussian from drifting into the one-electron region.
    m0_margin = max(0.5 * zero_peak_width, 2 * (zero_one_centers[1] - zero_one_centers[0]))
    if gain_seed is not None:
        # A user gain seed is a deliberate, confident guess for where the one-electron
        # peak sits. Pin mu_1 to a tight band of +/- _GAIN_SEED_FIT_MARGIN ADU around
        # the seeded location (zero_peak + gain_seed) so curve_fit cannot slide it down
        # onto the zero-peak tail (the old behaviour, where mu_1 drifted to the lower
        # bound ~one zero-peak width below the guess and reported a gain well under it).
        # Clamp into the histogram window so the bounds stay valid.
        m1_low = max(fit_left, one_peak_charge - _GAIN_SEED_FIT_MARGIN)
        m1_high = min(fit_right, one_peak_charge + _GAIN_SEED_FIT_MARGIN)
        if m1_high <= m1_low:
            # Seed landed at/outside the window edge: fall back to a minimal valid band.
            m1_low = max(fit_left, min(one_peak_charge, fit_right - 1e-6))
            m1_high = min(fit_right, m1_low + _GAIN_SEED_FIT_MARGIN)
    else:
        # No seed: auto-detect mode. Give the fit headroom past both edges of the gain
        # band so a real peak near an edge is centred on its bump rather than pinned to
        # it (a low-gain bump that sits just above the floor would otherwise pin at the
        # floor and read as the minimum gain or be rejected). The shoulder guard keeps
        # mu_1 a few zero-peak widths off the pedestal, so the one-electron Gaussian
        # cannot slide back onto the zero-peak shoulder -- but it is capped at ~one
        # zero-peak width below the detected bump: on a wide zero peak (4 widths >
        # the real gain) an uncapped guard would exclude the genuine peak from the
        # fit entirely, leaving curve_fit to park mu_1 on the far tail and fail the
        # acceptance band. Acceptance is restricted to [_MIN_GAIN_ADU, _MAX_GAIN_ADU]
        # after the fit (in calculate_noise_gain).
        shoulder_guard = min(zero_peak_charge + 4 * zero_peak_width,
                             one_peak_charge - zero_peak_width)
        m1_low = max(one_lo - _GAIN_FIT_MARGIN, shoulder_guard)
        m1_high = min(fit_right, one_hi + _GAIN_FIT_MARGIN)
        if m1_high <= m1_low:
            # A wide (noisy) zero peak can push the 4*zero_peak_width headroom past
            # one_hi, leaving m1_low above one_hi -- clamping m1_high toward the fixed
            # one_hi cannot repair that, since it doesn't depend on m1_low. Fall back to
            # a minimal valid band around the detected bump instead, clamped into the
            # fit window (same pattern as the seeded branch above).
            m1_low = max(fit_left, min(one_peak_charge, fit_right - 1e-6))
            m1_high = min(fit_right, m1_low + _GAIN_SEED_FIT_MARGIN)

    # Floor the shared width at a fraction of the zero-peak width: a real peak cannot
    # be narrower than the readout noise, so this stops curve_fit from collapsing the
    # Gaussians into a delta spike on a single noisy bin (the low-statistics failure
    # that otherwise yields a confident but bogus gain). popt layout is
    # (s, m0, m1, N0, N1): both peaks share the single width ``s``.
    sigma_floor = max(0.3 * zero_peak_width, (zero_one_centers[1] - zero_one_centers[0]) / 10, 1e-8)
    # Tie the one-electron amplitude to the data: N_1 is the fitted Gaussian's height
    # at mu_1, so it must match the observed height of the one-electron bump -- the
    # largest fit-histogram count in the allowed mu_1 band, after subtracting the zero
    # Gaussian's estimated contribution to each bin (with coarse bins, the raw count
    # of a bin near the band's lower edge is mostly zero-peak tail, which would force
    # a phantom N_1 and drag mu_1 down onto the shoulder to justify it). Bounding N_1
    # to that height +/- ~2*sqrt(height) (Poisson) stops the fit from deflating N_1
    # toward zero while the zero Gaussian stretches to absorb the bump; an empty band
    # leaves the lower bound at 0, so "no peak" stays reachable.
    band = (zero_one_centers >= m1_low) & (zero_one_centers <= m1_high)
    zero_est = max_zero_one_counts * np.exp(
        -(zero_one_centers - zero_peak_charge) ** 2 / (2 * zero_peak_width ** 2))
    one_peak_count = max(float(np.max((zero_one_counts - zero_est)[band])), 0.0) if band.any() else 0.0
    # Asymmetric Poisson band: the sqrt(counts) weighting systematically biases a
    # sparse peak's amplitude LOW (undershooting the bump top costs less than
    # overshooting the empty flank bins), so hold N_1 within 1 sigma below the
    # measured height but allow 2 sigma above.
    n1_low = max(0.0, one_peak_count - 0.5*np.sqrt(one_peak_count))
    n1_high = one_peak_count + 2.0 * np.sqrt(one_peak_count + 1.0)
    fit_bounds_low = [
        sigma_floor,
        max(fit_left, zero_peak_charge - m0_margin),
        m1_low,
        0,
        n1_low,
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
        n1_high,
    ]

    fit_bounds = (fit_bounds_low, fit_bounds_high)

    p0 = [
        zero_peak_width,
        zero_peak_charge,
        one_peak_charge,
        max_zero_one_counts,
        one_peak_count,
    ]
    p0 = _clip_to_bounds(p0, fit_bounds)

    return zero_one_counts, zero_one_edges, p0, fit_bounds, zero_one_range, found_one_peak


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


def get_zero_one_peaks_ext(data_ext,
                           n_bins=_ZERO_ONE_N_BINS, fit_bounds='default',
                           zero_one_test_range='auto',
                           window_left_scale=1.0, window_right_scale=1.0,
                           gain_seed=None):
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
            n_bins=n_bins,
            fit_bounds=fit_bounds,
            window_left_scale=window_left_scale,
            window_right_scale=window_right_scale,
            gain_seed=gain_seed_ext,
        )
        zero_one_counts_ext.append(zero_one_counts)
        zero_one_edges_ext.append(zero_one_edges)
        pedestals.append(pedestal)
        gains.append(gain)
        double_gauss_popts.append(double_gauss_popt)
        zero_one_ranges.append(zero_one_range)

    return zero_one_counts_ext, zero_one_edges_ext, pedestals, gains, double_gauss_popts, zero_one_ranges
