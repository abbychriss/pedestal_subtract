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
import json
import sys
from pathlib import Path

from .core import (
    get_fits,
    pedestal_subtract_ext_cached,
    get_zero_one_peaks_ext,
    plot_zero_one_peaks,
)


def _load_config(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def _config_default(config, key, fallback):
    return config[key] if key in config else fallback


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
    parser.add_argument("--pedestal_subtraction_axis", type=str,
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
                        default=_config_default(config, 'plot_zero_one_adu', False),
                        help="Plot the double-Gaussian fits in ADU.")
    parser.add_argument("--no-plot_zero_one_adu", dest="plot_zero_one_adu", action="store_false",
                        help="Disable the ADU-units plot.")
    parser.add_argument("--plot_zero_one_electrons", action="store_true",
                        default=_config_default(config, 'plot_zero_one_electrons', True),
                        help="Also produce the electron-units version of the fits.")
    parser.add_argument("--no-plot_zero_one_electrons", dest="plot_zero_one_electrons",
                        action="store_false", help="Disable the electron-units plot.")
    parser.add_argument("--plot_zero_one_individual", action="store_true",
                        default=_config_default(config, 'plot_zero_one_individual', False),
                        help="Plot one figure per extension.")
    parser.add_argument("--no-plot_zero_one_individual", dest="plot_zero_one_individual",
                        action="store_false", help="Disable individual per-extension figures.")
    parser.add_argument("--plot_zero_one_together", action="store_true",
                        default=_config_default(config, 'plot_zero_one_together', True),
                        help="Plot all extensions in a combined 2x2 subplot.")
    parser.add_argument("--no-plot_zero_one_together", dest="plot_zero_one_together",
                        action="store_false", help="Disable the combined 2x2 subplot.")
    parser.add_argument("--plot_zero_one_yscale", type=str,
                        default=_config_default(config, 'plot_zero_one_yscale', 'linear'),
                        help="Y-axis scale: 'linear' or 'log'.")
    parser.add_argument("--plot_zero_one_individual_figsize", nargs=2, type=float,
                        default=_config_default(config, 'plot_zero_one_individual_figsize', [7, 5]),
                        metavar=('W', 'H'), help="Figure size for individual plots.")
    parser.add_argument("--plot_zero_one_subplots_figsize", nargs=2, type=float,
                        default=_config_default(config, 'plot_zero_one_subplots_figsize', [10, 8]),
                        metavar=('W', 'H'), help="Figure size for the combined subplot.")
    parser.add_argument("--plot_zero_one_sharex", action="store_true",
                        default=_config_default(config, 'plot_zero_one_sharex', True),
                        help="Share the x-axis across the 2x2 subplot.")
    parser.add_argument("--no-plot_zero_one_sharex", dest="plot_zero_one_sharex",
                        action="store_false", help="Do not share the x-axis.")
    parser.add_argument("--plot_zero_one_sharey", action="store_true",
                        default=_config_default(config, 'plot_zero_one_sharey', True),
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
                        default=_config_default(config, 'show_titles', True),
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

    return parser.parse_args(argv)


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

    fig_path = Path(args.output_dir) if args.output_dir else file_path.parent / 'plots'
    if args.save_plots:
        fig_path.mkdir(parents=True, exist_ok=True)
        print(f'Plots will be saved to {fig_path}')

    print(f'Analyzing image: {file_path}\n')
    data_ext = get_fits(str(file_path))

    if args.do_pedestal_subtraction:
        data_ext = pedestal_subtract_ext_cached(
            data_ext,
            source_path=file_path,
            n_std_to_mask=args.n_std_to_mask,
            axis=args.pedestal_subtraction_axis,
            use_biweight_loc=args.use_biweight_loc,
            use_biweight_midvar=args.use_biweight_midvar,
            cache_dir=args.pedsub_cache_dir,
            force=args.force_pedsub,
            verbose=args.verbose,
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
        n=100,
        fit_bounds='default',
    )

    for ext, (pedestal, gain) in enumerate(zip(pedestals, gains)):
        noise = double_gauss_popts[ext][0]
        print(f'EXT {ext + 1}: pedestal = {pedestal:.4f} ADU, '
              f'noise = {noise:.4f} ADU, gain = {gain:.4f} ADU/e-')
    print()

    if args.plot_zero_one_adu or args.plot_zero_one_electrons:
        plot_zero_one_peaks(
            data_ext,
            zero_one_counts_ext,
            zero_one_edges_ext,
            pedestals,
            gains,
            double_gauss_popts,
            zero_one_ranges,
            individual_figsize=tuple(args.plot_zero_one_individual_figsize),
            subplots_figsize=tuple(args.plot_zero_one_subplots_figsize),
            xlim='default',
            ylim='default',
            additional_title=args.extra_plot_title if args.extra_plot_title else '',
            suptitle='Double-Gaussian Fit to Zero-One Electron Peaks',
            nimages=nimages,
            yscale=args.plot_zero_one_yscale,
            fontsize=12,
            n=100,
            do_plot_adu=args.plot_zero_one_adu,
            do_convert_to_electrons=args.plot_zero_one_electrons,
            plot_individual=args.plot_zero_one_individual,
            plot_together=args.plot_zero_one_together,
            sharex=args.plot_zero_one_sharex,
            sharey=args.plot_zero_one_sharey,
            show_titles=args.show_titles,
            save_plots=args.save_plots,
            show_plots=args.show_plots,
            fig_path=str(fig_path),
            file=file_path.name,
            dpi=350,
        )


if __name__ == "__main__":
    main()
