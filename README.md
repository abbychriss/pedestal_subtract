# pedestal_subtract

Standalone pedestal-subtraction + double-Gaussian fitting/plotting pipeline,
extracted from the `nonlinearity_studies` package.

## What it does

Given a FITS image with extensions 1–4 of pixel charge data, this package:

1. **Loads** the four extensions (`get_fits`).
2. **Pedestal-subtracts** each extension row-by-row (or column-wise), iteratively
   sigma-clipping to the zero-electron peak core. Results are cached to a
   `*.pedsub.fits` file next to the source so reruns with the same parameters skip
   the recompute (`pedestal_subtract_ext_cached`).
3. **Fits the zero/one-electron peaks** of each extension with a double Gaussian,
   yielding the pedestal, read noise, and gain (`get_zero_one_peaks_ext`).
4. **Plots** those double-Gaussian fits in ADU and/or electron units, individually
   per extension and/or as a combined 2×2 subplot (`plot_zero_one_peaks`).

## Installation

From this directory (`pedestal_subtract/`, the one containing `pyproject.toml`):

```bash
pip install -e .          # editable install (recommended while developing)
# or
pip install .
```

This installs the importable `pedestal_subtract` package and a `pedestal-subtract`
console command.

## Running

Once installed, run the console script from anywhere:

```bash
pedestal-subtract path/to/image.fits
```

Or run it as a module (works installed, or from this directory without installing):

```bash
python -m pedestal_subtract path/to/image.fits
```

Save plots without displaying them, and also produce the electron-units version:

```bash
python -m pedestal_subtract path/to/image.fits \
    --save_plots --no-show_plots --plot_zero_one_electrons -o ./plots
```

Drive everything from a JSON config (CLI flags override config values):

```bash
python -m pedestal_subtract -j config.json path/to/image.fits
```

See all options:

```bash
python -m pedestal_subtract --help
```

### Common options

| Flag | Default | Meaning |
|------|---------|---------|
| `--no-do_pedestal_subtraction` | (on) | Skip pedestal subtraction. |
| `--n_std_to_mask` | `1.5` | Sigma-clip width when estimating the pedestal. |
| `--pedestal_subtraction_axis` | `row` | `row`, `col`, `row_then_col`, or `col_then_row`. |
| `--force_pedsub` | off | Recompute, ignoring the on-disk cache. |
| `--pedsub_cache_dir DIR` | source dir | Where to write the `*.pedsub.fits` cache. |
| `-z` / `--plot_zero_one_adu` | on | Plot fits in ADU. |
| `--plot_zero_one_electrons` | off | Also plot in electron units. |
| `--plot_zero_one_yscale` | `linear` | `linear` or `log`. |
| `-s` / `--save_plots` | off | Save plots to `--output_dir`. |
| `--no-show_plots` | (shows) | Don't open figures interactively. |

## Config compatibility with `nonlinearity_studies`

The CLI reads the **same JSON config keys** as `nonlinearity_studies`, with the same
names, defaults, and types. You can point it at an existing nonlinearity config:

```bash
pedestal-subtract -j report_config.json
```

Keys outside this package's scope (e.g. `peak_finder_*`, `fit_range_right`,
`plot_nonlinearity_*`, `bin_factor`, `plot_all_peaks_*`) are silently ignored, and
`file_string` may be supplied in the config instead of on the command line — exactly
as in `nonlinearity_studies`. The keys this package consumes:

`file_string`, `do_pedestal_subtraction`, `n_std_to_mask`, `pedestal_subtraction_axis`,
`pedsub_cache_dir`, `force_pedsub`, `use_biweight_loc`, `use_biweight_midvar`,
`plot_zero_one_adu`, `plot_zero_one_electrons`, `plot_zero_one_individual`,
`plot_zero_one_together`, `plot_zero_one_yscale`, `plot_zero_one_individual_figsize`,
`plot_zero_one_subplots_figsize`, `plot_zero_one_sharex`, `plot_zero_one_sharey`,
`extra_plot_title`, `nimages`, `show_titles`, `save_plots`, `show_plots`,
`output_dir`, `verbose`.

## Using inside `nonlinearity_studies`

The exported functions are extracted verbatim from `nonlinearity_studies` and have
**identical signatures and bodies**, so they are a drop-in replacement for that
package's pedestal-subtraction + zero/one section:

```python
from pedestal_subtract import (
    pedestal_subtract_ext_cached,
    get_zero_one_peaks_ext,
    plot_zero_one_peaks,
)
```

## Using as a library

```python
from pedestal_subtract import (
    get_fits,
    pedestal_subtract_ext_cached,
    get_zero_one_peaks_ext,
    plot_zero_one_peaks,
)

data_ext = get_fits("image.fits")
data_ext = pedestal_subtract_ext_cached(data_ext, source_path="image.fits",
                                        n_std_to_mask=1.5, axis="row")
(counts, edges, pedestals, gains,
 popts, ranges) = get_zero_one_peaks_ext(data_ext, n=100)
```

## Requirements

- numpy
- scipy
- matplotlib
- astropy
- tqdm

## Layout

```
pedestal_subtract/          # project root (pip install from here)
├── pyproject.toml
├── README.md
└── pedestal_subtract/      # the importable package
    ├── __init__.py         # public API
    ├── __main__.py         # CLI entry point (python -m pedestal_subtract)
    └── core.py             # extracted analysis + plotting routines
```
