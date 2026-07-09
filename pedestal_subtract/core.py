"""
core — backwards-compatibility shim.

This module was split into focused submodules.  Everything is re-exported
here (including private, underscore-prefixed helpers) so existing imports
such as `from ...core import name` keep working unchanged.
"""

from .constants import (
    _ZERO_ONE_N_BINS,
    _GAIN_SEED_FIT_MARGIN,
    _ZERO_PEAK_LEFT_FIT_NSIGMA,
    _MIN_GAIN_ADU,
    _MAX_GAIN_ADU,
    _DARK_CURRENT_COLUMN,
    _DARK_CURRENT_METHODS,
    _EXTENSION_SUMMARY_FIELDS,
)
from .double_gauss_model import (
    double_gauss,
)
from .calibrate import (
    convert_to_electrons,
    calculate_noise_gain,
)
from .fit_zero_one import (
    _smooth_counts,
    _GAIN_FIT_MARGIN,
    _clamp_n_bins,
    _make_histogram,
    _estimate_peak_width,
    _clip_to_bounds,
    _auto_zero_one_setup,
    _one_electron_peak_is_real,
    get_zero_one_peaks_ext,
)
from .pedestal import (
    pedestal_subtract,
    _PEDSUB_ALGO_VERSION,
    _PEDSUB_HEADER_KEYS,
    _overscan_range_token,
    _overscan_key,
    _normalize_overscan_cols_ext,
    _pedsub_param_key,
    _pedsub_cache_path,
    _pedsub_header_matches,
    pedestal_subtract_ext_cached,
)
from .fits_io import (
    get_fits,
    get_exposure_time_days,
    get_pixel_binning,
    raw_pedestal_locations,
    _OVERSCAN_HEADER_KEYS,
    get_fits_header,
    overscan_cols_from_header,
    _value_for_extension,
    _scalar_for_extension,
)
from .dark_current import (
    _resolve_dark_current_methods,
    _single_electron_count_window,
    _single_electron_count_integral,
    _single_electron_count_weighted,
    calculate_dark_current,
)
from .plotting import (
    _SUBPLOTS_FIGSIZE,
    _INDIVIDUAL_FIGSIZE,
    _bar_heights,
    _fit_curve_x,
    _finish_fig,
    _zero_one_ylim,
    _double_gauss_popt_electrons,
    _electron_double_gauss_popt,
    _zero_one_label_adu,
    plot_zero_one_peaks,
    _dark_current_fit_label,
    plot_dark_current_zero_one,
    plot_charge_per_column,
)
from .summary import (
    build_extension_summary,
    _extension_summary_fieldnames,
    write_extension_summary_csv,
)

__all__ = [
    "_ZERO_ONE_N_BINS",
    "_GAIN_SEED_FIT_MARGIN",
    "_ZERO_PEAK_LEFT_FIT_NSIGMA",
    "_MIN_GAIN_ADU",
    "_MAX_GAIN_ADU",
    "_DARK_CURRENT_COLUMN",
    "_DARK_CURRENT_METHODS",
    "_EXTENSION_SUMMARY_FIELDS",
    "double_gauss",
    "convert_to_electrons",
    "calculate_noise_gain",
    "_smooth_counts",
    "_GAIN_FIT_MARGIN",
    "_clamp_n_bins",
    "_make_histogram",
    "_estimate_peak_width",
    "_clip_to_bounds",
    "_auto_zero_one_setup",
    "_one_electron_peak_is_real",
    "get_zero_one_peaks_ext",
    "pedestal_subtract",
    "_PEDSUB_ALGO_VERSION",
    "_PEDSUB_HEADER_KEYS",
    "_overscan_range_token",
    "_overscan_key",
    "_normalize_overscan_cols_ext",
    "_pedsub_param_key",
    "_pedsub_cache_path",
    "_pedsub_header_matches",
    "pedestal_subtract_ext_cached",
    "get_fits",
    "get_exposure_time_days",
    "get_pixel_binning",
    "raw_pedestal_locations",
    "_OVERSCAN_HEADER_KEYS",
    "get_fits_header",
    "overscan_cols_from_header",
    "_value_for_extension",
    "_scalar_for_extension",
    "_resolve_dark_current_methods",
    "_single_electron_count_window",
    "_single_electron_count_integral",
    "_single_electron_count_weighted",
    "calculate_dark_current",
    "_SUBPLOTS_FIGSIZE",
    "_INDIVIDUAL_FIGSIZE",
    "_bar_heights",
    "_fit_curve_x",
    "_finish_fig",
    "_zero_one_ylim",
    "_double_gauss_popt_electrons",
    "_electron_double_gauss_popt",
    "_zero_one_label_adu",
    "plot_zero_one_peaks",
    "_dark_current_fit_label",
    "plot_dark_current_zero_one",
    "plot_charge_per_column",
    "build_extension_summary",
    "_extension_summary_fieldnames",
    "write_extension_summary_csv",
]
