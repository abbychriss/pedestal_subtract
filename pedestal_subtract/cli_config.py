"""
cli_config — CLI config loading & argument-normalization helpers for the pedestal_subtract command line, split out of __main__.py.
"""
import argparse
import glob
import hashlib
import inspect
import json
import numpy as np
from analysis_tools import compute_hot_columns
from .core import plot_zero_one_peaks
from .constants import _DEFAULT_DARK_CURRENT_METHOD
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


def _is_int_token(v):
    """True when ``v`` is (or spells) a plain integer -- e.g. ``8``, ``'8'``, ``'-8'``.

    Used to tell the legacy bare-integer --fit_cols form (``8 615``) apart from the
    slice-string / keyword form (``'0:3200,3400:3500'``, ``'auto'``). ``bool`` is
    excluded so ``True``/``False`` are never mistaken for integers.
    """
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, np.integer)):
        return True
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in ('+', '-'):
            s = s[1:]
        return s.isdigit()
    return False


def _coerce_fit_cols(value):
    """Coerce a raw --fit_cols / config value into the canonical form consumed by
    ``_normalize_fit_cols``: ``None``, ``'auto'``, a ``[c0, c1]`` pair, a slice-string
    (``'lo:hi,lo:hi'``), or a per-extension list of any of those (``null`` = all columns).

    The CLI passes a list of string tokens (``type=str``, ``nargs='+'``). Three forms are
    accepted there and normalized here so both CLI and JSON feed ``_normalize_fit_cols``
    identically:

    * All tokens are bare integers -- the legacy form. Two ints (one range for every
      extension) or two per extension (e.g. 8 ints for 4 extensions) are grouped into
      ``[c0, c1]`` pairs. Raises ``ValueError`` on an odd int count.
    * A single non-integer token -- a slice-string or ``'auto'`` broadcast to every
      extension (e.g. ``'0:3200,3400:3500'``).
    * Several non-integer tokens -- one spec per extension; ``'null'``/``'none'`` (any
      case) becomes ``None`` (all columns for that extension).

    A JSON config passes the already-structured value (``'auto'``, a slice-string, a
    ``[c0, c1]`` pair, or a per-extension list), which is returned essentially unchanged.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        # Legacy bare-integer form: group the flat list into [c0, c1] pairs.
        if all(_is_int_token(v) for v in value):
            if len(value) % 2 != 0:
                raise ValueError(
                    f'--fit_cols needs an even number of integers (START STOP per '
                    f'extension) when given as bare integers; got {len(value)}')
            ints = [int(v) for v in value]
            pairs = [[ints[i], ints[i + 1]] for i in range(0, len(ints), 2)]
            return pairs[0] if len(pairs) == 1 else pairs
        # String / keyword form: map 'null'/'none' to None; unwrap a lone broadcast token.
        tokens = [None if (isinstance(v, str) and v.strip().lower() in ('null', 'none'))
                  else v for v in value]
        return tokens[0] if len(tokens) == 1 else tokens
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
    list here so both forms feed _resolve_dark_current_methods identically. A null
    value in the config (or the key being absent, which reaches here as None via the
    argparse default) selects the default method (_DEFAULT_DARK_CURRENT_METHOD).
    """
    if value is None:
        return [_DEFAULT_DARK_CURRENT_METHOD]
    if isinstance(value, str):
        return [value]
    return list(value)


def _as_endpoint(v):
    """A single slice endpoint as ``int`` or ``None`` (open end). Accepts ints, or
    strings that are an integer or blank/``none``/``null`` (an omitted endpoint)."""
    if v is None:
        return None
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return int(v)
    s = str(v).strip()
    if s == '' or s.lower() in ('none', 'null'):
        return None
    try:
        return int(s)
    except ValueError:
        raise ValueError(f'fit_cols endpoint {v!r} is not an integer')


def _parse_region(part):
    """Parse one ``'START:STOP'`` region string into a ``(lo, hi)`` endpoint pair.

    Both endpoints are optional (``'3400:'`` = to the end, ``':3200'`` = from the start),
    and negative values count from the right, matching Python half-open slice semantics.
    """
    if ':' not in part:
        raise ValueError(
            f'fit_cols region {part!r} must be "START:STOP" (a half-open column slice; '
            f'either end may be blank, e.g. "3400:" or ":3200")')
    lo_s, hi_s = part.split(':', 1)
    return (_as_endpoint(lo_s), _as_endpoint(hi_s))


def _parse_ext_spec(spec):
    """Parse one extension's fit_cols spec into ``None``, ``'auto'``, or a list of
    ``(lo, hi)`` keep-regions.

    ``spec`` may be ``None`` / ``'null'`` / ``'none'`` (keep all columns), ``'auto'``
    (keep all but the hot columns), a ``[c0, c1]`` pair (a single keep-region), or a
    comma-separated slice-string such as ``'0:3200,3400:3500'`` (a union of keep-regions).
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        s = spec.strip()
        if s.lower() in ('', 'null', 'none'):
            return None
        if s.lower() == 'auto':
            return 'auto'
        regions = [_parse_region(p.strip()) for p in s.split(',') if p.strip()]
        return regions or None
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return [(_as_endpoint(spec[0]), _as_endpoint(spec[1]))]
    raise ValueError(
        f'fit_cols entry must be null, "auto", a [START, STOP] pair, or a "lo:hi,lo:hi" '
        f'region string; got {spec!r}')


def _normalize_fit_cols(fit_cols, n_ext):
    """Resolve the canonical fit_cols value into one spec per extension.

    Each returned entry is ``None`` (keep all columns), ``'auto'`` (keep all but the hot
    columns), or a list of ``(lo, hi)`` half-open keep-regions. Accepts ``None`` (all
    columns everywhere), ``'auto'`` or a slice-string (broadcast to every extension), a
    single ``(c0, c1)`` pair (broadcast), or a length-``n_ext`` per-extension list whose
    entries are each ``None`` / ``'auto'`` / a slice-string / a ``(c0, c1)`` pair.
    ``_resolve_fit_col_masks`` later turns these specs into per-column keep-masks (which
    needs the data, for the ``'auto'`` hot-column computation). Mirrors
    ``core._normalize_overscan_cols_ext``.
    """
    if fit_cols is None:
        return [None] * n_ext
    # Explicit per-extension list (checked first so an n_ext-extension file isn't mistaken
    # for a single range): every entry must be None, a string, or a [c0, c1] 2-sequence.
    # A bare int is not a valid entry, so a [c0, c1] broadcast pair falls through below.
    if isinstance(fit_cols, (list, tuple)) and len(fit_cols) == n_ext and all(
            o is None or isinstance(o, str)
            or (isinstance(o, (list, tuple)) and len(o) == 2
                and all(x is None or (isinstance(x, (int, np.integer))
                                      and not isinstance(x, bool)) for x in o))
            for o in fit_cols):
        return [_parse_ext_spec(o) for o in fit_cols]
    # 'auto' or a slice-string -> broadcast to every extension.
    if isinstance(fit_cols, str):
        return [_parse_ext_spec(fit_cols)] * n_ext
    # A single (c0, c1) range -> broadcast to every extension.
    if isinstance(fit_cols, (list, tuple)) and len(fit_cols) == 2 and all(
            v is None or (isinstance(v, (int, np.integer)) and not isinstance(v, bool))
            for v in fit_cols):
        return [_parse_ext_spec(list(fit_cols))] * n_ext
    raise ValueError(
        f'fit_cols must be null, "auto", a [START, STOP] pair, a "lo:hi,lo:hi" region '
        f'string, or a length-{n_ext} per-extension list of those; got {fit_cols!r}')


def _resolve_fit_col_masks(fit_cols_ext, data_ext, n_std_to_mask):
    """Turn per-extension fit_cols specs into per-extension column keep-masks (or None).

    ``None`` keeps every column (returned as ``None`` so the caller can skip slicing).
    ``'auto'`` keeps every column except the hot ones (median charge >= ``n_std_to_mask``
    biweight SDs from the extension's location -- the same columns ``plot_charge_per_column``
    draws red), computed from ``data_ext`` (the raw, pre-pedestal-subtraction frame). A
    list of ``(lo, hi)`` regions keeps the union of those half-open column slices.
    """
    hot_ext = None
    masks = []
    for ext, (spec, data) in enumerate(zip(fit_cols_ext, data_ext)):
        if spec is None:
            masks.append(None)
            continue
        ncol = np.asarray(data).shape[1]
        if spec == 'auto':
            if hot_ext is None:
                hot_ext = compute_hot_columns(data_ext, n_std_to_mask)
            masks.append(~np.asarray(hot_ext[ext], dtype=bool))
            continue
        keep = np.zeros(ncol, dtype=bool)
        for lo, hi in spec:
            keep[lo:hi] = True
        masks.append(keep)
    return masks


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
    """argparse type for the single zero/one binning knob: a strict integer number of
    bins (>= 1) used for both the peak finder and the double-Gaussian fit/plot,
    independently of the fit-window scales or the gain guess."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {n})")
    return n


# Args that should NOT influence the run-identity hash: anything that only
# affects display/output (not the pedestal-subtracted data or the fit results).
_RUN_HASH_EXCLUDE = {
    'config', 'json', 'verbose', 'save_output', 'output_dir', 'show_plots',
    'pedsub_cache_dir', 'force_pedsub',
    'extra_plot_title', 'nimages',
    'plot_zero_one_adu', 'plot_zero_one_electrons',
    'plot_zero_one_individual', 'plot_zero_one_together',
    'plot_zero_one_yscale', 'plot_zero_one_xlim', 'plot_zero_one_ylim',
    'plot_zero_one_ylim_electrons',
    'plot_zero_one_individual_figsize', 'plot_zero_one_subplots_figsize',
    'plot_dark_current_figsize', 'plot_charge_per_column_figsize',
    'plot_charge_per_column_individual_figsize',
    'plot_zero_one_sharex', 'plot_zero_one_sharey', 'show_titles',
    'electron_fit_mode',
    'do_dark_current',
    'dark_current_method', 'dark_current_count_center', 'dark_current_count_nsigma',
    'exposure_time_s',
    'plot_dark_current', 'plot_charge_per_column_together', 'plot_charge_per_column_individual',
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
