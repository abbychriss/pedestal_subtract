"""
constants — split out of core.py.
"""

# Strict number of bins for the zero/one histograms. This is the single binning knob:
# it sets the histogram bin count for both the peak-finding step and the
# double-Gaussian fit/plot histogram, so every window (test range, peak-search range,
# fit window) is divided into this many bins regardless of how wide it is -- unlike the
# old bins-per-ADU density, the bin *width* now scales with the window width. Exposed
# to the user as `zero_one_n_bins` / `--zero_one_n_bins`.
_ZERO_ONE_N_BINS = 75


# When the user supplies a gain guess (`zero_one_gain_guess`), pin the fitted
# one-electron mean mu_1 to a tight window of +/- this many ADU around the guessed
# location (zero peak + guess). This stops curve_fit from sliding mu_1 down onto the
# zero-peak tail, which previously dragged the fitted gain well below the guess.
_GAIN_SEED_FIT_MARGIN = 0.05


# The double-Gaussian fit only uses histogram bins from this many right-half sigma
# BELOW the pedestal upward (the plot window is unaffected). Excludes the heavy
# non-Gaussian left tail of the zero peak, which would otherwise inflate the fitted
# sigma and pull mu_0 below zero. At 2.0 the fitted mu_0 lands on the pedestal
# (~0 after subtraction); smaller values track the clean right flank ever more
# closely but start pulling mu_0 above zero.
_ZERO_PEAK_LEFT_FIT_NSIGMA = 2.0


_MIN_GAIN_ADU = 0.5


_MAX_GAIN_ADU = 1.8


_DARK_CURRENT_COLUMN = {
    'count': 'dark_current_count_e_per_pix_day',
    'integrate': 'dark_current_integrate_e_per_pix_day',
    'weighted': 'dark_current_weighted_e_per_pix_day',
}


_DARK_CURRENT_METHODS = ('count', 'integrate', 'weighted')


# The dark-current method used when none is explicitly requested: a null
# ``dark_current_method`` config value, or the key being absent entirely, both
# resolve to this. Single source of truth for the default.
_DEFAULT_DARK_CURRENT_METHOD = 'weighted'


_EXTENSION_SUMMARY_FIELDS = ['ext', 'gain_adu_per_e', 'noise_adu', 'noise_e']
