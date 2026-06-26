# pedestal_subtract

Standalone pedestal-subtraction + double-Gaussian fitting/plotting pipeline,
extracted from the `nonlinearity_studies` package.

## What it does

Given a FITS image with extensions 1–4 of pixel charge data, this package:

1. **Loads** the four extensions (`get_fits`).
2. **Pedestal-subtracts** each extension row-by-row (or column-wise), iteratively
   sigma-clipping to the zero-electron peak core. Optionally, the per-row pedestal
   can be estimated from only the serial-overscan columns (`--use_overscan_only`)
   and still subtracted from the full frame. Results are cached to a
   `*.pedsub.fits` file next to the source so reruns with the same parameters skip
   the recompute (`pedestal_subtract_ext_cached`).
3. **Fits the zero/one-electron peaks** of each extension with a double Gaussian,
   yielding the pedestal, read noise, and gain (`get_zero_one_peaks_ext`).
4. **Plots** those double-Gaussian fits in ADU and/or electron units, individually
   per extension and/or as a combined 2×2 subplot (`plot_zero_one_peaks`).

## Installation

From this directory (`pedestal_subtract/`, the one containing `pyproject.toml`):

```bash
pip install .
```
or in editable mode
```bash
pip install -e .
```

This installs the importable `pedestal_subtract` package and a `pedestal-subtract` command.

## Running

Once installed, run the console script from anywhere:

```bash
pedestal-subtract path/to/image.fits
```

Or run it as a module (works installed, or from this directory without installing):

```bash
python -m pedestal_subtract path/to/image.fits
```

Save plots without displaying them, and also produce the electron version:

```bash
python -m pedestal_subtract path/to/image.fits \
    --save_plots --no-show_plots --plot_zero_one_electrons -o ./plots
```

If you want to configure the parameters yourself, you can use a JSON config (CLI flags override config values):

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
| `--use_overscan_only [EXT ...]` | off | Estimate the per-row pedestal from the overscan columns only (then subtract it from the full frame). Pass extension numbers (1–4) to apply to only those extensions, or the flag alone to apply to all; in the JSON config use `true`/`false` or a list like `[1, 3]`. Use `--no-use_overscan_only` to force it off. |
| `--overscan_cols START STOP` | from header | Column slice the pedestal is estimated from when `--use_overscan_only` is set. By default the overscan columns are computed from the FITS header (see below); this flag/config value is only used as a fallback when the header lacks the geometry keys. Negative endpoints count from the right; in the JSON config use `null` for an open-ended slice (e.g. `[-147, null]`). |
| `--zero_one_n_bins` | `100` | Number of bins spanning the zero/one fit window at window scale 1.0 (the count scales up automatically when the window is widened, keeping bin width constant). Integer ≥ 10. |
| `--zero_one_window_left_scale` | `1.0` | Scale the left half-width of the auto-computed zero/one fit window (≥ 1.0; > 1 widens toward lower charge). |
| `--zero_one_window_right_scale` | `1.0` | Scale the right half-width of the auto-computed zero/one fit window (≥ 1.0; > 1 widens past the one-electron peak). |
| `--force_pedsub` | off | Recompute, ignoring the on-disk cache. |
| `--pedsub_cache_dir DIR` | source dir | Where to write the `*.pedsub.fits` cache. |
| `-z` / `--plot_zero_one_adu` | on | Plot fits in ADU. |
| `--plot_zero_one_electrons` | off | Also plot in electron units. |
| `--plot_zero_one_yscale` | `linear` | `linear` or `log`. |
| `-s` / `--save_plots` | off | Save plots to `--output_dir`. |
| `--no-show_plots` | (shows) | Don't open figures interactively. |

When saving to disk, each run is stored in a unique subfolder named with a short
hash of the analysis-relevant config. The complete resolved config is also
written to `config.json` inside that folder for reproducibility.

### Overscan columns from the FITS header

When `--use_overscan_only` is set, the serial-overscan column slice is computed
directly from the CCD-geometry header keys rather than hard-coded. The detector
has `PRESCAN` inactive columns, then `PHYSCOL` physical columns, then the overscan;
a frame reads CCD columns `[NCOLPRE*NSBIN, (NCOL+NCOLPRE)*NSBIN]`, so the overscan
is every column past `PRESCAN + PHYSCOL`. In binned image columns that is the last

```
n = ((NCOL + NCOLPRE) * NSBIN - (PRESCAN + PHYSCOL)) // NSBIN
```

columns, i.e. the slice `(-n, None)`. The required keys are `PRESCAN`, `PHYSCOL`,
`NCOL`, `NCOLPRE`, and `NSBIN` (read from the first HDU that contains them). If any
key is missing, the pipeline falls back to the `--overscan_cols` flag / `overscan_cols`
config value. The helpers `get_fits_header` and `overscan_cols_from_header` are also
exported for library use.

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
`use_overscan_only`, `overscan_cols`,
`zero_one_n_bins`, `zero_one_window_left_scale`, `zero_one_window_right_scale`,
`pedsub_cache_dir`, `force_pedsub`, `use_biweight_loc`, `use_biweight_midvar`,
`plot_zero_one_adu`, `plot_zero_one_electrons`, `plot_zero_one_individual`,
`plot_zero_one_together`, `plot_zero_one_yscale`, `plot_zero_one_individual_figsize`,
`plot_zero_one_subplots_figsize`, `plot_zero_one_sharex`, `plot_zero_one_sharey`,
`extra_plot_title`, `nimages`, `show_titles`, `save_plots`, `show_plots`,
`output_dir`, `verbose`.

`use_overscan_only` and `overscan_cols` are additions specific to this package (not
present in `nonlinearity_studies`); older configs without them fall back to the
defaults (overscan-only off).

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
# Estimate the per-row pedestal from the overscan columns only, applied full-frame.
# overscan_cols may be a single (c0, c1) pair (all extensions) or a per-extension
# list of None/(c0, c1) -- here exts 1 and 3 use the overscan, exts 2 and 4 the full frame:
# data_ext = pedestal_subtract_ext_cached(data_ext, source_path="image.fits",
#                                         n_std_to_mask=1.5, axis="row",
#                                         overscan_cols=[(-147, None), None, (-147, None), None])
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
