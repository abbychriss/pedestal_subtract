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
import hashlib
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .core import (
    _PEDSUB_ALGO_VERSION,
    get_fits,
    pedestal_subtract_ext_cached,
    get_zero_one_peaks_ext,
    plot_zero_one_peaks,
)


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


def _load_config(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def _config_default(config, key, fallback):
    return config[key] if key in config else fallback


# Args that should NOT influence the run-identity hash: anything that only
# affects display/output (not the pedestal-subtracted data or the fit results).
_RUN_HASH_EXCLUDE = {
    'config', 'json', 'verbose', 'save_plots', 'output_dir', 'show_plots',
    'pedsub_cache_dir', 'force_pedsub',
    'extra_plot_title', 'nimages',
    'plot_zero_one_adu', 'plot_zero_one_electrons',
    'plot_zero_one_individual', 'plot_zero_one_together',
    'plot_zero_one_yscale', 'plot_zero_one_xlim', 'plot_zero_one_ylim',
    'plot_zero_one_individual_figsize', 'plot_zero_one_subplots_figsize',
    'plot_zero_one_sharex', 'plot_zero_one_sharey', 'show_titles',
}


# Display-only CLI args whose default lives in plot_zero_one_peaks's signature: each maps
# to the function parameter it feeds. The CLI leaves these None when unset, so this mapping
# is the single place that resolves them -- both for forwarding (main) and for recording the
# value actually used (the config snapshot).
_PLOT_ARG_TO_PARAM = {
    'plot_zero_one_individual_figsize': 'individual_figsize',
    'plot_zero_one_subplots_figsize': 'subplots_figsize',
    'plot_zero_one_yscale': 'yscale',
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
                        help="Path to the input FITS file (or set 'file_string' in the JSON config).")

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
                             "Defaults to the source FITS directory.")
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

    return args


def main(argv=None):
    args = init_argparse(argv)

    if args.extra_plot_title and not args.extra_plot_title.endswith((' ', '\n')):
        args.extra_plot_title = f'{args.extra_plot_title}: '

    if args.file_string is None:
        print('Error: file_string is required unless it is provided in the JSON config.')
        sys.exit(1)

    file_path = Path(args.file_string)
    if not file_path.is_absolute() and not file_path.exists():
        # Search the current tree for a matching filename.
        for found in Path('.').rglob(file_path.name):
            file_path = found
            break

    if not file_path.exists():
        print(f'Error: FITS file not found: {args.file_string}')
        sys.exit(1)

    default_fig_path = file_path.parent / 'plots'
    fig_path = Path(args.output_dir) if args.output_dir else default_fig_path

    # Hardcoded parameters that affect the fit results (not display). Defined here
    # so they are both passed to the fit and recorded in the run snapshot.
    fit_params = {
        'n': 100,
        'fit_bounds': 'default',
        'zero_one_test_range': 'auto',
    }

    run_hash = _run_hash(args)
    fig_path = fig_path / f'{file_path.stem}_{run_hash}'
    print(f'Run hash: {run_hash}')

    # Only materialize the run directory (and its config snapshot) when there is
    # something to save into it; interactive-only runs leave no directory behind.
    if args.save_plots:
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
        print(f'Plots will be saved to {fig_path}')

    print(f'Analyzing image: {file_path}\n')
    data_ext = get_fits(str(file_path))

    # Per-extension overscan setting: each selected extension estimates its per-row
    # pedestal from the overscan columns only (still subtracted from the full frame);
    # the rest estimate from the full frame.
    overscan_exts = _overscan_ext_indices(args.use_overscan_only, len(data_ext))
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

    # Infer the stitched-image count from the filename unless overridden.
    nimages = args.nimages
    if nimages is None:
        import re
        match = re.search(r'_(\d+)_stitched', str(file_path))
        nimages = int(match.group(1)) if match else 1

    (zero_one_counts_ext, zero_one_edges_ext, pedestals, gains,
     double_gauss_popts, zero_one_ranges) = get_zero_one_peaks_ext(
        data_ext,
        n=fit_params['n'],
        fit_bounds=fit_params['fit_bounds'],
        zero_one_test_range=fit_params['zero_one_test_range'],
    )

    for ext, (pedestal, gain) in enumerate(zip(pedestals, gains)):
        noise = double_gauss_popts[ext][0]
        print(f'EXT {ext + 1}: pedestal = {pedestal:.4f} ADU, '
              f'noise = {noise/gain:.4f} e-, gain = {gain:.4f} ADU/e-')
    print()

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


if __name__ == "__main__":
    main()
