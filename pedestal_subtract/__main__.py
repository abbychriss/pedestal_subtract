"""
Command-line entry point for the pedestal-subtraction + double-Gaussian pipeline.

Usage
-----
    python -m pedestal_subtract IMAGE.fits [options]
    python -m pedestal_subtract -j config.json IMAGE.fits

Loads a FITS file, (optionally) pedestal-subtracts every extension, fits the
zero/one-electron peaks with a double Gaussian, and plots those fits in ADU
and/or electron units.

Options can be supplied on the command line or via a JSON config (-j/--json);
command-line flags override config values, which override built-in defaults.
"""

import argparse
import glob
import hashlib
import inspect
import json
import sys
from datetime import datetime

import numpy as np
from pathlib import Path

from . import __version__
from .core import (
    _PEDSUB_ALGO_VERSION,
    _PEAKFIND_DENSITY,
    _SUBPLOTS_FIGSIZE,
    get_fits,
    get_fits_header,
    get_exposure_time_days,
    get_pixel_binning,
    raw_pedestal_locations,
    overscan_cols_from_header,
    pedestal_subtract_ext_cached,
    get_zero_one_peaks_ext,
    _scalar_for_extension,
    calculate_dark_current,
    plot_zero_one_peaks,
    plot_dark_current_zero_one,
    plot_charge_per_column,
    write_extension_summary_csv,
)
from .stitch_fits import stitch_fits


# Default serial-overscan column range (Python half-open slice) used to estimate
# the per-row pedestal when --use_overscan_only is set. Overridable via the
# --overscan_cols CLI flag or the 'overscan_cols' config key. Endpoints may be
# negative (from the right) or null/None for an open-ended slice; the default
# [-147, None] (i.e. [-147:]) is the last 147 columns.
OVERSCAN_COLS = (-147, None)


def _overscan_ext_indices(value, n_ext):
    """0-based extension indices that use overscan-only pedestal estimation.

    Parses the --use_overscan_only / config value, which may be:
      * False / None        -> no extensions (off)
      * True                -> all extensions
      * [] (bare CLI flag)  -> all extensions
      * [1, 3] (1-indexed)  -> only those extensions
    """
    if value is False or value is None:
        return set()
    if value is True:
        return set(range(n_ext))
    if isinstance(value, (list, tuple)):
        if len(value) == 0:                       # bare flag -> all extensions
            return set(range(n_ext))
        return {int(e) - 1 for e in value if 1 <= int(e) <= n_ext}
    return set(range(n_ext))                       # any other truthy scalar -> all


def _coerce_fit_cols(value):
    """Coerce a raw --fit_cols / config value into None, a [c0, c1] pair, or a list of
    per-extension None/[c0, c1] entries.

    The CLI passes a flat list of ints -- two (one range for every extension) or two per
    extension (e.g. 8 for 4 extensions) -- while a JSON config may instead give a single
    [c0, c1] pair or a per-extension list whose entries are [c0, c1] or null. A flat,
    even-length int list is grouped into pairs here so both forms feed
    ``_normalize_fit_cols`` identically. Raises ``ValueError`` on an odd int count.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(v, (int, np.integer)) for v in value):
        if len(value) % 2 != 0:
            raise ValueError(
                f'--fit_cols needs an even number of integers (START STOP per extension); '
                f'got {len(value)}')
        pairs = [[int(value[i]), int(value[i + 1])] for i in range(0, len(value), 2)]
        return pairs[0] if len(pairs) == 1 else pairs
    return value


def _coerce_gain_guess(value):
    """Coerce a raw --zero_one_gain_guess / config value into None, a single float, or a
    per-extension list of floats/None.

    The CLI passes a list of floats (``nargs='+'``): one value (a single guess for every
    extension) collapses to a scalar here, while several values are a per-extension list.
    A JSON config may instead give a single number or a per-extension list whose entries
    are numbers or null (null = auto-detect that extension). Returned unchanged otherwise.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return value[0]
        return list(value)
    return value


def _coerce_dark_current_method(value):
    """Coerce a raw --dark_current_method / config value into a list of method names.

    The CLI always yields a list (nargs='+'). A JSON config may instead give a bare
    string (old single-method format, e.g. "weighted"), wrapped into a single-element
    list here so both forms feed _resolve_dark_current_methods identically.
    """
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_fit_cols(fit_cols, n_ext):
    """Resolve fit_cols into one ``(c0, c1)``-or-None entry per extension.

    Accepts ``None`` (all columns everywhere), a single ``(c0, c1)`` pair (applied to
    every extension), or a length-``n_ext`` list whose entries are each ``None`` (all
    columns for that extension) or a ``(c0, c1)`` pair. Mirrors
    ``core._normalize_overscan_cols_ext``.
    """
    if fit_cols is None:
        return [None] * n_ext
    # Explicit per-extension list (checked first so an n_ext-extension file isn't mistaken
    # for a single range): every entry must be None or a 2-sequence.
    if isinstance(fit_cols, (list, tuple)) and len(fit_cols) == n_ext and all(
            o is None or (isinstance(o, (list, tuple)) and len(o) == 2) for o in fit_cols):
        return [tuple(o) if o is not None else None for o in fit_cols]
    # A single (c0, c1) range -> apply to every extension.
    if isinstance(fit_cols, (list, tuple)) and len(fit_cols) == 2 and all(
            v is None or isinstance(v, (int, np.integer)) for v in fit_cols):
        return [tuple(fit_cols)] * n_ext
    raise ValueError(
        f'fit_cols must be null, a [START, STOP] pair, or a length-{n_ext} list of '
        f'null/[START, STOP]; got {fit_cols!r}')


def _derive_data_path(file_path_str):
    """
    Derive the data path from the file path.

    Args:
        file_path_str: The file path string (can contain glob patterns like '*')

    Returns:
        tuple: (directory_path, image_pattern) where directory_path is the path to search
               and image_pattern is the file pattern to match
    """
    # Remove trailing slashes
    clean_path_str = file_path_str.rstrip('/')

    # Split path and extract the image pattern (last component)
    parts = clean_path_str.split('/')
    image_pattern = parts[-1]  # e.g., '*' or '*.fz'

    # Directory path is everything before the pattern
    directory_path = '/'.join(parts[:-1])  # e.g., 'examples/images/ten-images'

    # Convert to Path and make absolute if relative
    dir_path = Path(directory_path)
    if not dir_path.is_absolute():
        dir_path = Path.cwd() / dir_path

    return dir_path, image_pattern


def _wildcard_free_base(pattern_str):
    """Deepest leading directory of ``pattern_str`` that contains no glob wildcard.

    When the file pattern spans multiple directories (e.g. ``.../PD07*/cds_avg*.fz``),
    the stitched output must be anchored somewhere deterministic rather than inside a
    literal wildcard-named folder (the old ``out_path`` reused ``PD07*`` verbatim, which
    created an actual directory called ``PD07*``). This walks the leading path segments
    up to the first one containing a glob magic character, so a concrete single
    directory still gets its own ``stitched-fits`` subfolder, while a wildcard pattern
    falls back to the common parent above the wildcard. Always returns an absolute path.
    """
    base_parts = []
    for part in Path(pattern_str).parts:
        if glob.has_magic(part):
            break
        base_parts.append(part)
    base = Path(*base_parts) if base_parts else Path('.')
    if not base.is_absolute():
        base = Path.cwd() / base
    return base


def _load_config(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def _config_default(config, key, fallback):
    return config[key] if key in config else fallback


def _window_scale(value):
    """argparse type for the fit-window scale factors: a float >= 1.0.

    Values below 1 shrink the window inside the one-electron peak, which leaves
    the fit nothing to fit, so they are rejected up front."""
    f = float(value)
    if f < 1.0:
        raise argparse.ArgumentTypeError(f"must be >= 1.0 (got {f})")
    return f


def _n_bins(value):
    """argparse type for the zero/one histogram bin count: an integer >= 10
    (the double-Gaussian has 6 free parameters, so fewer bins is ill-posed)."""
    f = int(value)
    if f < 10:
        raise argparse.ArgumentTypeError(f"must be an integer >= 10 (got {f})")
    return f


def _peakfind_density(value):
    """argparse type for the peak-finding histogram density (bins per ADU): a
    number >= 1, used only to locate the zero/one peaks, independent of the
    fit/plot bin count set by --zero_one_n_bins."""
    f = float(value)
    if f < 1.0:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {f})")
    return f


# Args that should NOT influence the run-identity hash: anything that only
# affects display/output (not the pedestal-subtracted data or the fit results).
_RUN_HASH_EXCLUDE = {
    'config', 'json', 'verbose', 'save_plots', 'save_csv', 'output_dir', 'show_plots',
    'pedsub_cache_dir', 'force_pedsub',
    'extra_plot_title', 'nimages',
    'plot_zero_one_adu', 'plot_zero_one_electrons',
    'plot_zero_one_individual', 'plot_zero_one_together',
    'plot_zero_one_yscale', 'plot_zero_one_xlim', 'plot_zero_one_ylim',
    'plot_zero_one_individual_figsize', 'plot_zero_one_subplots_figsize',
    'plot_dark_current_figsize', 'plot_charge_per_column_figsize',
    'plot_zero_one_sharex', 'plot_zero_one_sharey', 'show_titles',
    'electron_fit_mode',
    'dark_current_method', 'dark_current_count_center', 'dark_current_count_nsigma',
    'exposure_time_s',
    'plot_dark_current', 'plot_charge_per_column',
}


# Display-only CLI args whose default lives in plot_zero_one_peaks's signature: each maps
# to the function parameter it feeds. The CLI leaves these None when unset, so this mapping
# is the single place that resolves them -- both for forwarding (main) and for recording the
# value actually used (the config snapshot).
_PLOT_ARG_TO_PARAM = {
    'plot_zero_one_individual_figsize': 'individual_figsize',
    'plot_zero_one_subplots_figsize': 'subplots_figsize',
    'plot_zero_one_yscale': 'yscale',
    'electron_fit_mode': 'electron_fit_mode',
    'plot_zero_one_individual': 'plot_individual',
    'plot_zero_one_together': 'plot_together',
    'plot_zero_one_sharex': 'sharex',
    'plot_zero_one_sharey': 'sharey',
    'show_titles': 'show_titles',
}


def _plot_defaults():
    """The signature defaults of plot_zero_one_peaks for the deferred display args."""
    params = inspect.signature(plot_zero_one_peaks).parameters
    return {arg: params[param].default for arg, param in _PLOT_ARG_TO_PARAM.items()}


def _resolved_args_dict(args):
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _effective_args_dict(args):
    """Like _resolved_args_dict, but fill deferred display args (left None on the CLI) with
    the plot_zero_one_peaks default actually used, so the snapshot records the real value."""
    out = _resolved_args_dict(args)
    for arg, default in _plot_defaults().items():
        if out.get(arg) is None:
            out[arg] = list(default) if isinstance(default, tuple) else default
    return out


def _run_hash(args, length=8):
    d = _resolved_args_dict(args)
    for key in _RUN_HASH_EXCLUDE:
        d.pop(key, None)
    payload = json.dumps(d, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha1(payload).hexdigest()[:length]


def init_argparse(argv=None):
    # Pre-parse so a JSON config can supply the defaults for every other argument.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-j", "--json", dest="config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config = _load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        prog="python -m pedestal_subtract",
        description="Pedestal-subtract FITS extensions and fit/plot the zero-one "
                    "electron peaks with a double Gaussian.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-j", "--json", type=str, default=pre_args.config,
                        help="Path to a JSON config file supplying argument defaults.")
    parser.add_argument("file_string", type=str, nargs='?',
                        default=_config_default(config, 'file_string', None),
                        help="Path to the input FITS file, or to a directory with per-image "
                             "FITS files (optionally with a glob pattern, e.g. "
                             "data/03-12-2026/avg*.fz) when --stitch_fits is set. "
                             "May also be set in the JSON config via 'file_string'.")

    # ----- Stitching -----
    parser.add_argument("-f", "--stitch_fits", action="store_true",
                        default=_config_default(config, 'stitch_fits', False),
                        help="Stitch the per-image FITS files matched by file_string into a "
                             "single combined FITS (by extension), then analyze that.")
    parser.add_argument("--no-stitch_fits", dest="stitch_fits", action="store_false",
                        help="Disable FITS stitching when enabled by the JSON config.")

    # ----- Pedestal subtraction -----
    parser.add_argument("--do_pedestal_subtraction", action="store_true",
                        default=_config_default(config, 'do_pedestal_subtraction', True),
                        help="Pedestal-subtract along one or two axes before fitting.")
    parser.add_argument("--no-do_pedestal_subtraction", dest="do_pedestal_subtraction",
                        action="store_false", help="Skip pedestal subtraction.")
    parser.add_argument("--n_std_to_mask", type=float,
                        default=_config_default(config, 'n_std_to_mask', 1.5),
                        help="Std devs from the per-line center to keep when estimating the pedestal.")
    parser.add_argument("--pedsub_max_iter", type=int,
                        default=_config_default(config, 'pedsub_max_iter', 5),
                        help="Max sigma-clip iterations when estimating each per-line pedestal.")
    parser.add_argument("-a", "--axis", "--pedestal_subtraction_axis", type=str,
                        dest="pedestal_subtraction_axis",
                        default=_config_default(config, 'pedestal_subtraction_axis', 'row'),
                        choices=['row', 'col', 'column', 'row_then_col', 'col_then_row'],
                        help="Axis to compute the pedestal across.")
    parser.add_argument("--pedsub_cache_dir", type=str,
                        default=_config_default(config, 'pedsub_cache_dir', None),
                        help="Directory for the pedestal-subtracted FITS cache. "
                             "Defaults to a 'cache/' folder beside the source FITS file.")
    parser.add_argument("--force_pedsub", action="store_true",
                        default=_config_default(config, 'force_pedsub', False),
                        help="Recompute pedestal subtraction, ignoring any cache.")
    parser.add_argument("--no-force_pedsub", dest="force_pedsub", action="store_false",
                        help="Use the cache when its params match (default).")
    parser.add_argument("--use_overscan_only", nargs='*', type=int, metavar='EXT',
                        default=_config_default(config, 'use_overscan_only', False),
                        help="Estimate the per-row pedestal from the overscan columns "
                             "(see --overscan_cols) only, then subtract it from the full "
                             "frame. Give extension numbers (1-4) to apply to only those "
                             "extensions, or pass the flag alone to apply to all. In the "
                             "JSON config use true/false or a list like [1, 3]. "
                             "(Put the FITS path before this flag, or in the config, so it "
                             "isn't read as an extension number.)")
    parser.add_argument("--no-use_overscan_only", dest="use_overscan_only",
                        action="store_const", const=False,
                        help="Estimate the pedestal from the full frame for all extensions (default).")
    parser.add_argument("--overscan_cols", nargs=2, type=int,
                        default=_config_default(config, 'overscan_cols', list(OVERSCAN_COLS)),
                        metavar=('START', 'STOP'),
                        help="Column range (Python half-open slice START:STOP) the per-row "
                             "pedestal is estimated from when --use_overscan_only is set. "
                             "Negative endpoints count from the right. Default: the last 147 "
                             "columns ([-147:]).")
    parser.add_argument("--use_biweight_loc", action="store_true",
                        default=_config_default(config, 'use_biweight_loc', True),
                        help="Use Tukey biweight location instead of a simple mean.")
    parser.add_argument("--no-use_biweight_loc", dest="use_biweight_loc", action="store_false",
                        help="Use a simple mean for the per-line center.")
    parser.add_argument("--use_biweight_midvar", action="store_true",
                        default=_config_default(config, 'use_biweight_midvar', True),
                        help="Use Tukey biweight midvariance instead of std dev.")
    parser.add_argument("--no-use_biweight_midvar", dest="use_biweight_midvar", action="store_false",
                        help="Use std dev for the per-line scale.")

    # ----- Fit window -----
    parser.add_argument("--zero_one_n_bins", type=_n_bins,
                        default=_config_default(config, 'zero_one_n_bins', 100),
                        help="Number of bins spanning the zero/one fit window at window "
                             "scale 1.0 (the count scales up automatically when the window "
                             "is widened, keeping bin width constant). Integer >= 10. Default 100.")
    parser.add_argument("--zero_one_window_left_scale", type=_window_scale,
                        default=_config_default(config, 'zero_one_window_left_scale', 1.0),
                        help="Scale the left half-width of the auto-computed zero/one fit "
                             "window (>=1.0; >1 widens toward lower charge). Default 1.0.")
    parser.add_argument("--zero_one_window_right_scale", type=_window_scale,
                        default=_config_default(config, 'zero_one_window_right_scale', 1.0),
                        help="Scale the right half-width of the auto-computed zero/one fit "
                             "window (>=1.0; >1 widens past the one-electron peak). Default 1.0.")
    parser.add_argument("--zero_one_peakfind_density", type=_peakfind_density,
                        default=_config_default(config, 'zero_one_peakfind_density', _PEAKFIND_DENSITY),
                        help="Bins-per-ADU of the internal histograms used to LOCATE the zero/one "
                             "peaks (separate from --zero_one_n_bins, which sets the fit/plot bins). "
                             "Raise for finer detection, lower to aggregate sparse low-statistics "
                             f"hits. Number >= 1. Default {_PEAKFIND_DENSITY}.")
    parser.add_argument("--fit_cols", nargs='+', type=int,
                        default=_config_default(config, 'fit_cols', None),
                        metavar='COL',
                        help="Restrict the zero/one fit (and the plotted histograms) to image "
                             "columns (a Python half-open slice START:STOP). Pass two ints "
                             "(START STOP) to apply one range to every extension, or two per "
                             "extension (e.g. 8 ints for 4 extensions) for per-extension ranges. "
                             "Negative endpoints count from the right. Applied after pedestal "
                             "subtraction, so the pedestal still uses the full frame/overscan. Omit "
                             "to use all columns. In the JSON config use a [START, STOP] pair, or a "
                             "per-extension list of [START, STOP]/null (null = all columns for that "
                             "extension; null endpoints like [256, null] give an open-ended slice).")
    parser.add_argument("--zero_one_gain_guess", nargs='+', type=float,
                        default=_config_default(config, 'zero_one_gain_guess', None),
                        metavar='GAIN',
                        help="Seed for the one-electron peak location -- a guess for the gain "
                             "(ADU/e-) -- used to initialize the double-Gaussian fit instead of "
                             "auto-detecting the one-electron bump. Pass one value to apply it to "
                             "every extension, or one per extension (e.g. 4 values for 4 "
                             "extensions). In the JSON config use a single number or a "
                             "per-extension list of numbers/null (null = auto-detect that "
                             "extension). Omit to auto-detect for all. Values must be > 0.")

    # ----- Plotting -----
    parser.add_argument("-z", "--plot_zero_one_adu", action="store_true",
                        default=_config_default(config, 'plot_zero_one_adu', True),
                        help="Plot the double-Gaussian fits in ADU.")
    parser.add_argument("--no-plot_zero_one_adu", dest="plot_zero_one_adu", action="store_false",
                        help="Disable the ADU-units plot.")
    parser.add_argument("--plot_zero_one_electrons", action="store_true",
                        default=_config_default(config, 'plot_zero_one_electrons', True),
                        help="Also produce the electron-units version of the fits.")
    parser.add_argument("--no-plot_zero_one_electrons", dest="plot_zero_one_electrons",
                        action="store_false", help="Disable the electron-units plot.")
    parser.add_argument("--plot_dark_current", action="store_true",
                        default=_config_default(config, 'plot_dark_current', True),
                        help="Plot the electron-unit zero/one distributions with the "
                             "dark-current count window marked (vertical lines, when the "
                             "'count' method is used) and a legend of the shared sigma, "
                             "gain and dark current.")
    parser.add_argument("--no-plot_dark_current", dest="plot_dark_current",
                        action="store_false", help="Disable the dark-current window plot.")
    parser.add_argument("--plot_charge_per_column", action="store_true",
                        default=_config_default(config, 'plot_charge_per_column', False),
                        help="Plot the median charge per column for each extension on the raw "
                             "(pre-pedestal-subtraction) data, with columns whose median is "
                             ">= n_std_to_mask biweight-SDs from the location flagged red. "
                             "Every column is plotted; the columns excluded by --fit_cols are "
                             "shaded light grey.")
    parser.add_argument("--no-plot_charge_per_column", dest="plot_charge_per_column",
                        action="store_false", help="Disable the charge-per-column plot.")
    parser.add_argument("--electron_fit_mode", type=str, choices=['transform', 'refit'],
                        default=_config_default(config, 'electron_fit_mode', None),
                        help="How to obtain the electron-unit double-Gaussian curve: "
                             "'transform' (default) analytically rescales the converged ADU fit "
                             "(exact; mu_0=0 / mu_1=1 by construction, no refit); 'refit' fits "
                             "the double Gaussian again directly to the electron-unit histogram, "
                             "letting the peaks/widths re-optimise in electron space.")
    # Display-only options below default to None when unset (no CLI flag, not in config)
    # so plot_zero_one_peaks's signature defaults stay the single source of truth; main()
    # forwards only the ones that were actually set.
    parser.add_argument("--plot_zero_one_individual", action="store_true",
                        default=_config_default(config, 'plot_zero_one_individual', None),
                        help="Plot one figure per extension.")
    parser.add_argument("--no-plot_zero_one_individual", dest="plot_zero_one_individual",
                        action="store_false", help="Disable individual per-extension figures.")
    parser.add_argument("--plot_zero_one_together", action="store_true",
                        default=_config_default(config, 'plot_zero_one_together', None),
                        help="Plot all extensions in a combined 2x2 subplot.")
    parser.add_argument("--no-plot_zero_one_together", dest="plot_zero_one_together",
                        action="store_false", help="Disable the combined 2x2 subplot.")
    parser.add_argument("--plot_zero_one_yscale", type=str,
                        default=_config_default(config, 'plot_zero_one_yscale', None),
                        help="Y-axis scale: 'linear' or 'log'.")
    # argparse runs `type` over string defaults, which would choke on the 'default'
    # / 'none' keywords, so default to None here and fall back to the config below.
    parser.add_argument("--plot_zero_one_xlim", nargs=2, type=float, default=None,
                        metavar=('LOW', 'HIGH'),
                        help="X-axis limits for the zero-one plots. "
                             "Omit (or set 'default') to use the fit range; 'none' to autoscale.")
    parser.add_argument("--plot_zero_one_ylim", nargs=2, type=float, default=None,
                        metavar=('LOW', 'HIGH'),
                        help="Y-axis limits for the ADU zero-one plots. "
                             "Omit (or set 'default') for the auto range; 'none' to autoscale.")
    parser.add_argument("--plot_zero_one_ylim_electrons", nargs=2, type=float, default=None,
                        metavar=('LOW', 'HIGH'),
                        help="Y-axis limits for the electron-units zero-one plots. Separate from "
                             "--plot_zero_one_ylim since the electron peaks are taller (smaller sigma). "
                             "Omit (or set 'default') for the auto range; 'none' to autoscale.")
    # No fallback here: when neither the CLI nor the config sets a size, leave it None
    # so plot_zero_one_peaks's own signature default is the single source of truth.
    parser.add_argument("--plot_zero_one_individual_figsize", nargs=2, type=float,
                        default=_config_default(config, 'plot_zero_one_individual_figsize', None),
                        metavar=('W', 'H'), help="Figure size for individual plots.")
    parser.add_argument("--plot_zero_one_subplots_figsize", nargs=2, type=float,
                        default=_config_default(config, 'plot_zero_one_subplots_figsize', None),
                        metavar=('W', 'H'), help="Figure size for the combined subplot.")
    parser.add_argument("--plot_dark_current_figsize", nargs=2, type=float,
                        default=_config_default(config, 'plot_dark_current_figsize', None),
                        metavar=('W', 'H'),
                        help=f"Figure size for the dark-current window plot. "
                             f"Default: the shared subplots size {_SUBPLOTS_FIGSIZE}.")
    parser.add_argument("--plot_charge_per_column_figsize", nargs=2, type=float,
                        default=_config_default(config, 'plot_charge_per_column_figsize', None),
                        metavar=('W', 'H'),
                        help=f"Figure size for the charge-per-column plot. "
                             f"Default: the shared subplots size {_SUBPLOTS_FIGSIZE}.")
    parser.add_argument("--plot_zero_one_sharex", action="store_true",
                        default=_config_default(config, 'plot_zero_one_sharex', None),
                        help="Share the x-axis across the 2x2 subplot.")
    parser.add_argument("--no-plot_zero_one_sharex", dest="plot_zero_one_sharex",
                        action="store_false", help="Do not share the x-axis.")
    parser.add_argument("--plot_zero_one_sharey", action="store_true",
                        default=_config_default(config, 'plot_zero_one_sharey', None),
                        help="Share the y-axis across the 2x2 subplot.")
    parser.add_argument("--no-plot_zero_one_sharey", dest="plot_zero_one_sharey",
                        action="store_false", help="Do not share the y-axis.")

    # ----- Titles / output -----
    parser.add_argument("--extra_plot_title", type=str,
                        default=_config_default(config, 'extra_plot_title', '') or '',
                        help="Prefix prepended to the plot title.")
    parser.add_argument("--nimages", type=int,
                        default=_config_default(config, 'nimages', None),
                        help="Number of stitched images (shown in the title). Inferred from "
                             "the filename when not set.")
    parser.add_argument("--show_titles", action="store_true",
                        default=_config_default(config, 'show_titles', None),
                        help="Draw plot titles.")
    parser.add_argument("--no-show_titles", dest="show_titles", action="store_false",
                        help="Hide plot titles.")
    parser.add_argument("-s", "--save_plots", action="store_true",
                        default=_config_default(config, 'save_plots', False),
                        help="Save plots to the output directory.")
    parser.add_argument("--save_csv", action="store_true",
                        default=_config_default(config, 'save_csv', False),
                        help="Save a per-extension gain/noise summary as "
                             "extension_summary.csv in the output directory.")
    parser.add_argument("--no-save_csv", dest="save_csv", action="store_false",
                        help="Do not write the extension_summary.csv file.")
    parser.add_argument("--dark_current_method", nargs='+', type=str,
                        default=_config_default(config, 'dark_current_method', ['all']),
                        metavar='METHOD',
                        help="How to count single-electron events for the dark current "
                             "(electrons/pixel/day). One or more of 'count' (pixels "
                             "within one peak sigma of 1 e-), 'integrate' (area under "
                             "the fitted one-electron Gaussian, a conservative upper "
                             "bound), 'weighted' (the one-electron amplitude fraction "
                             "N1/(N1+N0)), or 'all' (every method; default). Pass "
                             "multiple values to compute several at once, e.g. "
                             "--dark_current_method count weighted; each writes its "
                             "own CSV column so they can be compared.")
    parser.add_argument("--dark_current_count_center", type=str,
                        choices=['one_electron', 'mu1'],
                        default=_config_default(config, 'dark_current_count_center', 'one_electron'),
                        help="For the 'count' method, where the +/- window is "
                             "centred: 'one_electron' (default) centres on the ideal 1 e- "
                             "charge (pedestal + gain); 'mu1' centres on the fitted "
                             "one-electron peak mean.")
    parser.add_argument("--dark_current_count_nsigma", type=float,
                        default=_config_default(config, 'dark_current_count_nsigma', 1.0),
                        help="Half-width of the 'count' window, in units of the fitted "
                             "peak width sigma (default 1.0 = +/- 1 sigma). The single "
                             "shared sigma (the pedestal width) is well-determined even at "
                             "low statistics. The windowed count is divided by the Gaussian "
                             "fraction erf(nsigma/sqrt(2)) (0.6827 at +/- 1 sigma) to "
                             "estimate the full one-electron count. "
                             "Use a value < 1 to shrink the window and cut zero-peak tail "
                             "contamination, which dominates the count at low dark current.")
    parser.add_argument("--exposure_time_s", type=float,
                        default=_config_default(config, 'exposure_time_s', None),
                        help="Manually specify the exposure time in SECONDS, overriding the "
                             "FITS-header value (DATEEND - DATEINI) used for the dark current. "
                             "Useful when a file's header is missing DATEINI/DATEEND. Converted "
                             "to days internally. Default: read from the header.")
    parser.add_argument("--show_plots", action="store_true",
                        default=_config_default(config, 'show_plots', True),
                        help="Display plots interactively.")
    parser.add_argument("--no-show_plots", dest="show_plots", action="store_false",
                        help="Do not display plots (useful with --save_plots).")
    parser.add_argument("-o", "--output_dir", type=str,
                        default=_config_default(config, 'output_dir', None),
                        help="Directory for saved plots. Defaults to a 'plots' folder "
                             "next to the source FITS.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        default=_config_default(config, 'verbose', False),
                        help="Print cache / progress messages.")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false",
                        help="Quiet the cache / progress messages.")

    args = parser.parse_args(argv)

    # Fall back to the config keyword (or 'default') when no CLI limits were given.
    if args.plot_zero_one_xlim is None:
        args.plot_zero_one_xlim = _config_default(config, 'plot_zero_one_xlim', 'default')
    if args.plot_zero_one_ylim is None:
        args.plot_zero_one_ylim = _config_default(config, 'plot_zero_one_ylim', 'default')
    if args.plot_zero_one_ylim_electrons is None:
        args.plot_zero_one_ylim_electrons = _config_default(config, 'plot_zero_one_ylim_electrons', 'default')

    # argparse's `type` validates CLI strings but not config-supplied defaults, so
    # re-check the window scales here to also reject values < 1 coming from the JSON.
    for _scale_arg in ('zero_one_window_left_scale', 'zero_one_window_right_scale'):
        if getattr(args, _scale_arg) < 1.0:
            parser.error(f"{_scale_arg} must be >= 1.0 (got {getattr(args, _scale_arg)})")
    if int(args.zero_one_n_bins) < 10:
        parser.error(f"zero_one_n_bins must be an integer >= 10 (got {args.zero_one_n_bins})")
    args.zero_one_n_bins = int(args.zero_one_n_bins)
    if float(args.zero_one_peakfind_density) < 1.0:
        parser.error(f"zero_one_peakfind_density must be >= 1 (got {args.zero_one_peakfind_density})")
    args.zero_one_peakfind_density = float(args.zero_one_peakfind_density)

    # argparse's choices only validate CLI strings, so re-check a config-supplied value.
    if args.electron_fit_mode not in (None, 'transform', 'refit'):
        parser.error(f"electron_fit_mode must be 'transform' or 'refit' (got {args.electron_fit_mode!r})")
    args.dark_current_method = _coerce_dark_current_method(args.dark_current_method)
    _bad_dc_methods = [m for m in args.dark_current_method
                       if m not in ('count', 'integrate', 'weighted', 'all')]
    if _bad_dc_methods:
        parser.error(f"dark_current_method must be one or more of 'count', 'integrate', "
                     f"'weighted', or 'all' (got invalid value(s) {_bad_dc_methods!r})")
    if not args.dark_current_method:
        parser.error("dark_current_method must include at least one of 'count', "
                     "'integrate', 'weighted', or 'all'")
    if args.dark_current_count_center not in ('one_electron', 'mu1'):
        parser.error(f"dark_current_count_center must be 'one_electron' or 'mu1' "
                     f"(got {args.dark_current_count_center!r})")
    if float(args.dark_current_count_nsigma) <= 0:
        parser.error(f"dark_current_count_nsigma must be > 0 "
                     f"(got {args.dark_current_count_nsigma})")
    args.dark_current_count_nsigma = float(args.dark_current_count_nsigma)

    # Group a flat CLI int list into column pairs (and validate the count) up front, so
    # the stored/snapshotted value is the canonical pair / per-extension form.
    try:
        args.fit_cols = _coerce_fit_cols(args.fit_cols)
    except ValueError as e:
        parser.error(str(e))

    # Collapse a single gain seed to a scalar (per-extension list otherwise) and check
    # every supplied value is positive -- the gain (ADU/e-) is always > 0. argparse's
    # `type` only validates CLI floats, so the per-entry check also covers config values.
    args.zero_one_gain_guess = _coerce_gain_guess(args.zero_one_gain_guess)
    _gg = args.zero_one_gain_guess
    _gg_values = _gg if isinstance(_gg, list) else ([] if _gg is None else [_gg])
    for v in _gg_values:
        if v is not None and float(v) <= 0:
            parser.error(f"zero_one_gain_guess values must be > 0 (got {v})")

    return args


def main(argv=None):
    args = init_argparse(argv)

    if args.extra_plot_title and not args.extra_plot_title.endswith((' ', '\n')):
        args.extra_plot_title = f'{args.extra_plot_title}\n'

    if args.file_string is None:
        print('Error: file_string is required unless it is provided in the JSON config.')
        sys.exit(1)

    file_path = Path(args.file_string)

    # When not stitching, file_string must resolve to an existing FITS file (with a
    # tree-wide search fallback for a bare relative name). When stitching, file_string is
    # instead a directory / glob of per-image FITS files, so this check is skipped.
    if not args.stitch_fits:
        if glob.has_magic(args.file_string):
            # A glob pattern (e.g. '..._VDD-20p5_*.fz') must be expanded -- otherwise the
            # literal '*' path never exists and the check below reports "file not found".
            # The single-file (non-stitch) path analyzes one image, so an ambiguous match
            # is an error: narrow the pattern or use --stitch_fits to combine the files.
            matches = sorted(glob.glob(args.file_string))
            if not matches:
                print(f'Error: no FITS file matches the pattern: {args.file_string}')
                sys.exit(1)
            if len(matches) > 1:
                print(f'Error: file_string matched {len(matches)} files; narrow the pattern '
                      'or use --stitch_fits to combine them:\n  ' + '\n  '.join(matches))
                sys.exit(1)
            file_path = Path(matches[0])
        elif not file_path.is_absolute() and not file_path.exists():
            # Search the current tree for a matching filename.
            for found in Path('.').rglob(file_path.name):
                file_path = found
                break

        if not file_path.exists():
            print(f'Error: FITS file not found: {args.file_string}')
            sys.exit(1)

    # Optionally stitch the per-image FITS files into a single combined FITS, then analyze
    # that. If file_string already points inside a 'stitched-fits' directory, reuse the
    # existing stitched file instead of re-stitching.
    stitched_dir_name = 'stitched-fits'
    if args.stitch_fits:
        input_dir, input_pattern = _derive_data_path(args.file_string)

        if stitched_dir_name in input_dir.parts:
            stitched_match = next(input_dir.glob(input_pattern), None)
            if stitched_match is None:
                print('\nError: no files found matching the specified stitched FITS pattern.')
                sys.exit(1)
            file_path = stitched_match
        else:
            # Anchor the output at the deepest wildcard-free directory of the pattern, so a
            # multi-directory glob (e.g. PD07*/...) writes one combined file to a single
            # 'stitched-fits' folder instead of a literal 'PD07*' directory. out_path is
            # absolute, so it overrides stitch_fits's own file_path/out_path join.
            out_path = (_wildcard_free_base(args.file_string) / stitched_dir_name).resolve()
            stitched_file = stitch_fits(
                input_dir.parent,
                directory=input_dir.name,
                image=input_pattern,
                out_path=out_path,
                print_header=args.verbose,
            )
            if stitched_file is None:
                sys.exit(1)
            file_path = Path(stitched_file)

    # Plots live next to the source image, or beside the 'stitched-fits' parent when stitched.
    if stitched_dir_name in file_path.parts:
        stitched_fits_idx = file_path.parts.index(stitched_dir_name)
        base_path = Path(*file_path.parts[:stitched_fits_idx])
        default_fig_path = base_path / 'plots'
    else:
        default_fig_path = file_path.parent / 'plots'
    fig_path = Path(args.output_dir) if args.output_dir else default_fig_path

    # Hardcoded parameters that affect the fit results (not display). Defined here
    # so they are both passed to the fit and recorded in the run snapshot.
    fit_params = {
        'n': args.zero_one_n_bins,
        'fit_bounds': 'default',
        'zero_one_test_range': 'auto',
        'window_left_scale': args.zero_one_window_left_scale,
        'window_right_scale': args.zero_one_window_right_scale,
        'peakfind_density': args.zero_one_peakfind_density,
        'gain_seed': args.zero_one_gain_guess,
    }

    run_hash = _run_hash(args)
    fig_path = fig_path / f'{file_path.stem}_{run_hash}'
    print(f'Run hash: {run_hash}')

    # Only materialize the run directory (and its config snapshot) when there is
    # something to save into it; interactive-only runs leave no directory behind.
    if args.save_plots or args.save_csv:
        fig_path.mkdir(parents=True, exist_ok=True)
        config_snapshot_path = fig_path / 'config.json'
        snapshot = {
            'run_hash': run_hash,
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'package_version': __version__,
            'pedsub_algo_version': _PEDSUB_ALGO_VERSION,
            'fit_params': fit_params,
            'args': _effective_args_dict(args),
        }
        with open(config_snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, sort_keys=True, default=str)
        print(f'Config snapshot saved to {config_snapshot_path}')
        if args.save_plots:
            print(f'Plots will be saved to {fig_path}')

    print(f'Analyzing image: {file_path}\n')
    data_ext = get_fits(str(file_path))

    # Capture the pedestal (zero-peak) location from the RAW data, before any
    # pedestal subtraction, so the reported baseline is the physical pedestal level
    # rather than the ~0 baseline left after subtraction.
    raw_pedestals = raw_pedestal_locations(data_ext)
    # Keep the full raw frame (pedestal subtraction below rebinds data_ext) for the
    # charge-per-column diagnostic, which is meaningful on the un-subtracted data.
    raw_data_ext = data_ext

    # Per-extension fit-column slice and stitched-image count, resolved up front so the
    # raw charge-per-column diagnostic below (and the fit later) both use them.
    fit_cols_ext = _normalize_fit_cols(args.fit_cols, len(data_ext))
    nimages = args.nimages
    if nimages is None:
        import re
        match = re.search(r'_(\d+)_stitched', str(file_path))
        nimages = int(match.group(1)) if match else 1

    # Charge per column is a RAW-data diagnostic and must run BEFORE pedestal subtraction:
    # the subtraction levels (and its sigma-clip masks) the very anomalous columns this
    # plot is meant to reveal. It is therefore also the first figure shown.
    if args.plot_charge_per_column:
        # Own figsize when configured, else the shared subplots default (13x9).
        charge_figsize = (tuple(args.plot_charge_per_column_figsize)
                          if args.plot_charge_per_column_figsize is not None else _SUBPLOTS_FIGSIZE)
        plot_charge_per_column(
            raw_data_ext,
            n_std_to_mask=args.n_std_to_mask,
            fit_cols_ext=fit_cols_ext,
            figsize=charge_figsize,
            additional_title=args.extra_plot_title if args.extra_plot_title else '',
            show_titles=args.show_titles if args.show_titles is not None else True,
            nimages=nimages,
            verbose=args.verbose,
            save_plots=args.save_plots,
            show_plots=args.show_plots,
            fig_path=str(fig_path),
            file=file_path.name,
            dpi=350,
        )

    # Per-extension overscan setting: each selected extension estimates its per-row
    # pedestal from the overscan columns only (still subtracted from the full frame);
    # the rest estimate from the full frame.
    overscan_exts = _overscan_ext_indices(args.use_overscan_only, len(data_ext))
    if overscan_exts:
        # Prefer the overscan columns computed from the CCD-geometry header keys
        # (PRESCAN, PHYSCOL, NCOL, NCOLPRE, NSBIN); fall back to --overscan_cols /
        # the config value only when the header lacks those keys.
        overscan_range = overscan_cols_from_header(get_fits_header(file_path))
        if overscan_range is None:
            overscan_range = tuple(args.overscan_cols)
            if args.verbose:
                print(f'Overscan columns: header keys not found; using configured {overscan_range}')
        elif args.verbose:
            print(f'Overscan columns from header: {overscan_range} (last {-overscan_range[0]} columns)')
    else:
        overscan_range = tuple(args.overscan_cols)
    overscan_cols = [overscan_range if i in overscan_exts else None
                     for i in range(len(data_ext))]

    if args.do_pedestal_subtraction:
        data_ext = pedestal_subtract_ext_cached(
            data_ext,
            source_path=file_path,
            n_std_to_mask=args.n_std_to_mask,
            axis=args.pedestal_subtraction_axis,
            use_biweight_loc=args.use_biweight_loc,
            use_biweight_midvar=args.use_biweight_midvar,
            max_iter=args.pedsub_max_iter,
            cache_dir=args.pedsub_cache_dir,
            force=args.force_pedsub,
            verbose=args.verbose,
            overscan_cols=overscan_cols,
        )

    # Restrict the columns used for the zero/one fit (and the plotted histograms), if
    # configured. Done after pedestal subtraction so the per-row pedestal (and any
    # overscan estimate) still sees the full frame. The fit flattens each extension, so
    # this only changes which pixels enter the zero/one histogram, not the geometry.
    # One (c0, c1) range may be applied to every extension, or one per extension.
    # (fit_cols_ext was resolved above, before pedestal subtraction.)
    if any(fc is not None for fc in fit_cols_ext):
        data_ext = [
            np.asarray(d)[:, fc[0]:fc[1]] if fc is not None else d
            for d, fc in zip(data_ext, fit_cols_ext)
        ]
        if args.verbose:
            print(f'Restricting the fit columns per extension: {fit_cols_ext}')

    if args.verbose and fit_params['gain_seed'] is not None:
        # Show the per-extension gain seeds actually applied (resolved with the same
        # helper the fit uses), so a single value is reported as the list it broadcasts to.
        gain_seeds_ext = [_scalar_for_extension(fit_params['gain_seed'], ext, len(data_ext))
                          for ext in range(len(data_ext))]
        print(f'Using guess for gains: {gain_seeds_ext}')

    (zero_one_counts_ext, zero_one_edges_ext, pedestals, gains,
     double_gauss_popts, zero_one_ranges) = get_zero_one_peaks_ext(
        data_ext,
        n=fit_params['n'],
        fit_bounds=fit_params['fit_bounds'],
        zero_one_test_range=fit_params['zero_one_test_range'],
        window_left_scale=fit_params['window_left_scale'],
        window_right_scale=fit_params['window_right_scale'],
        peakfind_density=fit_params['peakfind_density'],
        gain_seed=fit_params['gain_seed'],
    )

    # Exposure time (days). A manually configured exposure_time_s (in seconds) overrides
    # the header value -- e.g. for a file whose FITS header is missing DATEINI/DATEEND;
    # otherwise read DATEEND - DATEINI from the header (same for a stitched image, since
    # each pixel was exposed for one image's duration).
    if args.exposure_time_s is not None:
        exposure_days = args.exposure_time_s / 86400
        # The manual value wins, but if the header also carries a valid exposure that
        # disagrees, flag it -- a mismatch usually means the wrong value was entered.
        header_days = get_exposure_time_days(file_path)
        if not np.isnan(header_days) and not np.isclose(
                header_days * 86400, args.exposure_time_s, rtol=1e-3, atol=0.5):
            print(f'Warning: manual exposure_time_s ({args.exposure_time_s:.1f} s) conflicts '
                  f'with the FITS-header DATEEND - DATEINI ({header_days * 86400:.1f} s); '
                  f'using the manual value.\n')
        if args.verbose:
            print(f'Exposure time (manual exposure_time_s): {args.exposure_time_s:.1f} s '
                  f'-> {exposure_days:.6f} days\n')
    else:
        exposure_days = get_exposure_time_days(file_path)
        if np.isnan(exposure_days):
            print('Warning: DATEINI/DATEEND not found in the FITS headers; dark current '
                  'will be NaN. Set exposure_time_s (seconds) in the config to supply it '
                  'manually.\n')
        elif args.verbose:
            total_s = exposure_days * 86400
            hours = int(total_s // 3600)
            minutes = int((total_s % 3600) // 60)
            print(f'Exposure time: {exposure_days:.6f} days / '
                  f'{hours}hr:{minutes:02d}min / {total_s:.1f} s\n')

    # Pixel binning (NPBIN x NSBIN): each image pixel sums that many physical CCD
    # pixels, so the dark current (per physical pixel) divides by the binned factor.
    npbin, nsbin = get_pixel_binning(file_path)
    if npbin is None or nsbin is None:
        print(f'Warning: NPBIN/NSBIN not found in the FITS headers (got NPBIN={npbin}, '
              f'NSBIN={nsbin}); assuming no binning (factor 1) for the dark current.\n')
    pixel_binning = (npbin or 1) * (nsbin or 1)
    if args.verbose:
        print(f'Pixel binning: NPBIN={npbin}, NSBIN={nsbin} -> {pixel_binning} '
              f'physical pixels per image pixel\n')

    # Per-extension dark current (electrons / physical pixel / day), by the chosen method(s).
    dark_current_rows = calculate_dark_current(
        data_ext, pedestals, gains, double_gauss_popts, zero_one_edges_ext,
        exposure_days, method=args.dark_current_method,
        count_center=args.dark_current_count_center,
        count_nsigma=args.dark_current_count_nsigma,
        pixel_binning=pixel_binning,
    )

    for ext, gain in enumerate(gains):
        noise = double_gauss_popts[ext][0]
        raw_pedestal = raw_pedestals[ext]
        if np.isnan(gain):
            # No trustworthy one-electron peak: gain (hence noise in e-) is undefined.
            print(f'EXT {ext + 1}: pedestal (raw) = {raw_pedestal:.4f} ADU, '
                  f'noise = {noise:.4f} ADU, no 1 e- peak found (gain undefined)')
        else:
            print(f'EXT {ext + 1}: pedestal (raw) = {raw_pedestal:.4f} ADU, '
                  f'noise = {noise/gain:.4f} e-, gain = {gain:.4f} ADU/e-')
        dc = dark_current_rows[ext]
        dc_parts = []
        if 'dark_current_count_e_per_pix_day' in dc:
            dc_parts.append(f"count (±{args.dark_current_count_nsigma:g}σ0) = "
                            f"{dc['dark_current_count_e_per_pix_day']:.4g}")
        if 'dark_current_integrate_e_per_pix_day' in dc:
            dc_parts.append(f"integrate = {dc['dark_current_integrate_e_per_pix_day']:.4g}")
        if 'dark_current_weighted_e_per_pix_day' in dc:
            dc_parts.append(f"weighted = {dc['dark_current_weighted_e_per_pix_day']:.4g}")
        if dc_parts:
            print(f"         dark current (e-/pix/day): {', '.join(dc_parts)}")
    print()

    if args.save_csv:
        summary_csv_path = fig_path / 'extension_summary.csv'
        write_extension_summary_csv(summary_csv_path, gains, double_gauss_popts,
                                    dark_current_rows=dark_current_rows,
                                    exposure_days=exposure_days,
                                    raw_pedestals=raw_pedestals)
        print(f'Saved per-extension summary to {summary_csv_path}\n')

    if args.plot_zero_one_adu or args.plot_zero_one_electrons:
        # Forward only the display options the user/config actually set (non-None); the rest
        # fall through to plot_zero_one_peaks's signature defaults, the single source of truth.
        plot_overrides = {
            param: getattr(args, arg) for arg, param in _PLOT_ARG_TO_PARAM.items()
            if getattr(args, arg) is not None
        }
        for key in ('individual_figsize', 'subplots_figsize'):
            if key in plot_overrides:
                plot_overrides[key] = tuple(plot_overrides[key])

        plot_zero_one_peaks(
            data_ext,
            zero_one_counts_ext,
            zero_one_edges_ext,
            pedestals,
            gains,
            double_gauss_popts,
            zero_one_ranges,
            **plot_overrides,
            xlim=args.plot_zero_one_xlim,
            ylim=args.plot_zero_one_ylim,
            ylim_electrons=args.plot_zero_one_ylim_electrons,
            additional_title=args.extra_plot_title if args.extra_plot_title else '',
            suptitle='Double-Gaussian Fit to Zero-One Electron Peaks',
            nimages=nimages,
            fontsize=12,
            n=fit_params['n'],
            do_plot_adu=args.plot_zero_one_adu,
            do_convert_to_electrons=args.plot_zero_one_electrons,
            save_plots=args.save_plots,
            show_plots=args.show_plots,
            fig_path=str(fig_path),
            file=file_path.name,
            dpi=350,
        )

    if args.plot_dark_current:
        plot_dark_current_zero_one(
            data_ext,
            zero_one_counts_ext,
            zero_one_edges_ext,
            pedestals,
            gains,
            double_gauss_popts,
            zero_one_ranges,
            dark_current_rows,
            count_center=args.dark_current_count_center,
            count_nsigma=args.dark_current_count_nsigma,
            electron_fit_mode=args.electron_fit_mode or 'transform',
            nimages=nimages,
            figsize=(tuple(args.plot_dark_current_figsize)
                     if args.plot_dark_current_figsize is not None else _SUBPLOTS_FIGSIZE),
            yscale=args.plot_zero_one_yscale or 'log',
            additional_title=args.extra_plot_title if args.extra_plot_title else '',
            show_titles=args.show_titles if args.show_titles is not None else True,
            save_plots=args.save_plots,
            show_plots=args.show_plots,
            fig_path=str(fig_path),
            file=file_path.name,
            dpi=350,
        )


if __name__ == "__main__":
    main()
