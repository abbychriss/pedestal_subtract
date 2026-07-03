"""
constants — split out of core.py.
"""

_PEAKFIND_DENSITY = 10


_MIN_GAIN_ADU = 0.5


_MAX_GAIN_ADU = 1.5


_DARK_CURRENT_COLUMN = {
    'count': 'dark_current_count_e_per_pix_day',
    'integrate': 'dark_current_integrate_e_per_pix_day',
    'weighted': 'dark_current_weighted_e_per_pix_day',
}


_DARK_CURRENT_METHODS = ('count', 'integrate', 'weighted')


_EXTENSION_SUMMARY_FIELDS = ['ext', 'gain_adu_per_e', 'noise_adu', 'noise_e']
