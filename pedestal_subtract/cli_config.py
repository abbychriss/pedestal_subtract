"""
cli_config — CLI config loading & argument-normalization helpers for the pedestal_subtract command line, split out of __main__.py.
"""
import argparse
import glob
import hashlib
import inspect
import json
import numpy as np
from .core import plot_zero_one_peaks
from pathlib import Path

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
    return set(range(n_ext))


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
