"""
calibrate — split out of core.py.
"""
import numpy as np
from scipy.optimize import curve_fit

from .constants import (
    _MAX_GAIN_ADU,
    _MIN_GAIN_ADU,
    _ZERO_ONE_N_BINS,
    _ZERO_PEAK_LEFT_FIT_NSIGMA,
)
from .double_gauss_model import double_gauss

def convert_to_electrons(data, pedestal, gain, flatten=True):
    if flatten:
        data = np.array(data).flatten()
    data_electrons = (data - pedestal) / gain  # Subtract pedestal (mean ADU of zero electron peak) and divide by gain
    return data_electrons


def calculate_noise_gain(data, zero_one_test_range='auto', n_bins=_ZERO_ONE_N_BINS,
                         fit_bounds='default',
                         window_left_scale=1.0, window_right_scale=1.0,
                         gain_seed=None):

    # Imported here rather than at module top to break the calibrate <-> fit_zero_one
    # import cycle: fit_zero_one imports calculate_noise_gain, and these zero/one setup
    # helpers live alongside it. Keeping calibrate import-time leaf-only avoids the cycle.
    from .fit_zero_one import _auto_zero_one_setup, _clip_to_bounds, _estimate_one_peak_amplitude

    data = np.array(data).flatten()
    data = data[np.isfinite(data)]

    if data.size == 0:
        raise ValueError('Input data contains no finite values')

    zero_one_counts, zero_one_edges, p0, auto_fit_bounds, zero_one_range, _found_one_peak = _auto_zero_one_setup(
        data,
        zero_one_test_range,
        n_bins=n_bins,
        window_left_scale=window_left_scale,
        window_right_scale=window_right_scale,
        gain_seed=gain_seed,
    )
    zero_one_centers = 0.5 * (zero_one_edges[:-1] + zero_one_edges[1:])

    if isinstance(fit_bounds, str) and fit_bounds in ('default', 'auto'):
        fit_bounds = auto_fit_bounds
    else:
        p0 = _clip_to_bounds(p0, fit_bounds)

    # Fit only from ~2 right-half sigma below the pedestal upward. The zero peak can
    # carry a heavy NON-Gaussian left tail (a pedestal-subtraction artefact: at 3 sigma
    # the left side holds ~1.7x the Gaussian prediction while the right side is clean),
    # and a symmetric Gaussian fed that tail inflates sigma and drags mu_0 below zero.
    # All the physics (the pedestal core and the one-electron region) lives from the
    # core rightward, so the left tail is excluded from the FIT; the histogram/plot
    # window is untouched, which also makes the anomalous tail visible against the
    # fitted curve. The width scale is the robust right-half sigma (median of positive
    # deviations from the median, /0.6745), immune to both the left tail and binning.
    median = float(np.median(data))
    above_median = data[data > median] - median
    sigma_right = float(np.median(above_median)) / 0.6745 if above_median.size else 0.0
    fit_mask = zero_one_centers >= median - _ZERO_PEAK_LEFT_FIT_NSIGMA * sigma_right
    if sigma_right <= 0 or not np.isfinite(sigma_right) or fit_mask.sum() < 6:
        fit_mask = np.ones_like(zero_one_centers, dtype=bool)

    # Two-stage fit. Stage 1 is an UNWEIGHTED least-squares fit of all 5 parameters:
    # unweighted LSQ is dominated by the ~1e4-count zero-peak core (the one-electron
    # region costs it almost nothing), which is exactly what we want here -- sigma,
    # mu_0, and N0 come out set by the bright, well-determined pedestal, not perturbed
    # by whatever the one-electron peak's amplitude/location happen to do. We keep
    # s, mu_0, and N0 from this stage and fix all three going into stage 2.
    popt_stage1, _ = curve_fit(double_gauss, zero_one_centers[fit_mask],
                               zero_one_counts[fit_mask], p0=p0,
                               maxfev=20000, bounds=fit_bounds)
    s_fit, m0_fit, N0_fit = popt_stage1[0], popt_stage1[1], popt_stage1[3]

    # Stage 2 refits only mu_1 and N1, with sigma, mu_0, and N0 FIXED at the stage-1
    # values, using a Poisson weight (sigma = sqrt(counts)) so the low-count
    # one-electron region gets to pull mu_1/N1 in proportion to its statistical error
    # rather than being steamrolled by the zero-peak core.
    #
    # N1's seed/bounds are recomputed here (rather than reusing fit_zero_one's
    # pre-fit p0/bounds) from the now-fitted mu_0/sigma: the "tallest bin >= 3 sigma
    # to the right of mu_0" cutoff needs the real sigma to actually clear the zero
    # peak's tail. Anchoring it to the pre-fit heuristic estimate (which can run
    # narrower than the true sigma) leaves the cutoff sitting inside the zero peak's
    # shoulder, which is still substantial there -- picking that up as "N1" overstates
    # the one-electron amplitude by orders of magnitude.
    m1_low, m1_high = fit_bounds[0][2], fit_bounds[1][2]
    one_peak_count, n1_low, n1_high = _estimate_one_peak_amplitude(
        zero_one_centers, zero_one_counts, m0_fit, s_fit, N0_fit)

    def _double_gauss_fixed_width_mean_n0(x, m1, N1):
        return double_gauss(x, s_fit, m0_fit, m1, N0_fit, N1)

    p0_stage2 = _clip_to_bounds([p0[2], one_peak_count], ([m1_low, n1_low], [m1_high, n1_high]))
    bounds_stage2 = ([m1_low, n1_low], [m1_high, n1_high])
    popt_stage2, pcov = curve_fit(_double_gauss_fixed_width_mean_n0, zero_one_centers[fit_mask],
                                  zero_one_counts[fit_mask], p0=p0_stage2,
                                  maxfev=20000, bounds=bounds_stage2,
                                  sigma=np.sqrt(np.maximum(zero_one_counts[fit_mask], 1.0)),
                                  absolute_sigma=True)
    m1_fit, N1_fit = popt_stage2
    popt = np.array([s_fit, m0_fit, m1_fit, N0_fit, N1_fit])

    # Extract pedestal and noise from the fit.
    pedestal = m0_fit  # Pedestal is mean of the zero-electron peak
    noise = s_fit      # Noise is the shared peak width (std of the zero-electron peak)

    # Decide whether the one-electron Gaussian is a real peak or just the fitter
    # using the second Gaussian to absorb the (non-Gaussian) tail of the zero peak
    # (see _one_electron_peak_is_real). "No peak" therefore fires only when there
    # is genuinely no localized excess -- a real gain in the allowed band passes.
    # A fitted separation outside [_MIN_GAIN_ADU, _MAX_GAIN_ADU] is rejected as
    # not a real 0->1 step.
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
