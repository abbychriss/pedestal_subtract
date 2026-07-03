"""
fit_zero_one — split out of core.py.
"""
import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks

from .calibrate import calculate_noise_gain
from .constants import _MAX_GAIN_ADU, _MIN_GAIN_ADU, _PEAKFIND_DENSITY
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
