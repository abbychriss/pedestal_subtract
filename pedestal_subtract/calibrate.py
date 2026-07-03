"""
calibrate — split out of core.py.
"""
import numpy as np
from scipy.optimize import curve_fit

from .constants import _MAX_GAIN_ADU, _MIN_GAIN_ADU, _PEAKFIND_DENSITY
from .double_gauss_model import double_gauss

def convert_to_electrons(data, pedestal, gain, flatten=True):
    if flatten:
        data = np.array(data).flatten()
    data_electrons = (data - pedestal) / gain  # Subtract pedestal (mean ADU of zero electron peak) and divide by gain
    return data_electrons


def calculate_noise_gain(data, zero_one_test_range='auto', n=100, fit_bounds='default',
                         window_left_scale=1.0, window_right_scale=1.0,
                         peakfind_density=_PEAKFIND_DENSITY, gain_seed=None):

    # Imported here rather than at module top to break the calibrate <-> fit_zero_one
    # import cycle: fit_zero_one imports calculate_noise_gain, and these zero/one setup
    # helpers live alongside it. Keeping calibrate import-time leaf-only avoids the cycle.
    from .fit_zero_one import _auto_zero_one_setup, _clip_to_bounds

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
