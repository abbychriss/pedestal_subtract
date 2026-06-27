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
| `--zero_one_n_bins` | `100` | Number of bins spanning the zero/one fit window at window scale 1.0, used for **both** the double-Gaussian fit and the plot (the count scales up automatically when the window is widened, keeping bin width constant). Integer ≥ 10. |
| `--zero_one_window_left_scale` | `1.0` | Scale the left half-width of the auto-computed zero/one fit window (≥ 1.0; > 1 widens toward lower charge). |
| `--zero_one_window_right_scale` | `1.0` | Scale the right half-width of the auto-computed zero/one fit window (≥ 1.0; > 1 widens past the one-electron peak). |
| `--zero_one_peakfind_density` | `10` | Bins-per-ADU of the internal histograms used to **locate** the zero/one peaks (independent of `--zero_one_n_bins`, which sets the fit/plot bins). Raise for finer detection; lower to aggregate sparse hits in low-statistics images. The histogram is floored at 50 bins, so very low values all behave the same. Number ≥ 1. |
| `--fit_cols COL ...` | all columns | Restrict the zero/one fit (and the plotted histograms) to image columns (a Python half-open slice `START:STOP`). Pass two ints (`START STOP`) to apply one range to **every** extension, or two **per** extension (e.g. 8 ints for 4 extensions) for per-extension ranges. Negative endpoints count from the right. In the JSON config use a `[START, STOP]` pair, or a per-extension list of `[START, STOP]`/`null` (e.g. `[[10, -10], null, [10, -10], null]`, where `null` keeps all columns for that extension; `null` endpoints like `[256, null]` give an open-ended slice). Applied **after** pedestal subtraction, so the per-row pedestal / overscan estimate still uses the full frame. Omit to use every column. |
| `--zero_one_gain_guess GAIN ...` | auto-detect | Seed for the one-electron peak location — a guess for the gain (ADU/e⁻) — used to **initialize** the double-Gaussian fit instead of auto-detecting the one-electron bump. The seed places `μ₁` at `μ₀ + gain` and sizes the fit window / amplitude guess from it; the post-fit acceptance band (gain ∈ [0.5, 1.5] ADU/e⁻) is unchanged. Pass one value to apply it to **every** extension, or one **per** extension (e.g. 4 values for 4 extensions). In the JSON config use a single number or a per-extension list of numbers/`null` (e.g. `[1.05, null, 0.95, 1.1]`, where `null` auto-detects that extension). Values must be > 0. Omit to auto-detect for all. |
| `--force_pedsub` | off | Recompute, ignoring the on-disk cache. |
| `--pedsub_cache_dir DIR` | source dir | Where to write the `*.pedsub.fits` cache. |
| `-z` / `--plot_zero_one_adu` | on | Plot fits in ADU. |
| `--plot_zero_one_electrons` | off | Also plot in electron units. |
| `--electron_fit_mode` | `transform` | How the electron-unit curve is obtained. `transform` analytically rescales the converged ADU fit (exact; μ₀ = 0 / μ₁ = 1 by construction, no refit). `refit` fits the double Gaussian again directly to the electron-unit histogram, letting the peaks/widths re-optimise in electron space (widths kept positive, amplitudes non-negative, means free; falls back to the transform if the refit fails). Only affects the electron-units plot. |
| `--plot_zero_one_yscale` | `linear` | `linear` or `log`. |
| `-s` / `--save_plots` | off | Save plots to `--output_dir`. |
| `--save_csv` | off | Write a per-extension `extension_summary.csv` (columns `ext`, `gain_adu_per_e`, `noise_adu`, `noise_e`) into the run's output folder. `noise_adu` is the zero-peak width (σ₀) in ADU; `noise_e` is that width divided by the gain. For an extension with no one-electron peak, `gain_adu_per_e` and `noise_e` are `nan`, but `noise_adu` is still reported (it does not depend on the gain). |
| `--no-show_plots` | (shows) | Don't open figures interactively. |

When saving to disk, each run is stored in a unique subfolder named with a short
hash of the analysis-relevant config. The complete resolved config is also
written to `config.json` inside that folder for reproducibility.

### Low-statistics one-electron peaks

The double-Gaussian fit is designed to behave sensibly when an extension has only
a handful of single-electron hits. Two physical priors drive this:

- **The gain (ADU/e⁻) is always > 1**, and is capped here at **1.5**. The
  one-electron peak is therefore constrained to sit in `[pedestal + 1.0,
  pedestal + 1.5]`, which is what stops the fit from sliding the second Gaussian
  down onto the zero-peak shoulder (the classic low-statistics failure). The
  band edges are the `_MIN_GAIN_ADU` / `_MAX_GAIN_ADU` constants in `core.py`.
- **A real peak is a localized excess, not a tail.** A one-electron peak is
  reported only when, after subtracting the fitted zero-electron Gaussian, the
  residual in a band around the peak rises above the shoulder just inside it (the
  valley toward the pedestal) and carries a statistically significant number of
  pixels. A strict local maximum is *not* required, so a weak peak that is only a
  few flat-topped bins still counts; but a heavy yet **monotonic** tail of the
  zero peak — which only falls off — does **not** produce a spurious gain.

When no genuine peak is found, the gain is reported as undefined (`NaN`), the
console prints `no 1 e- peak found`, and the plot shows only the single
zero-electron Gaussian. Detection resolution is tunable separately from the
fit/plot binning via `--zero_one_peakfind_density`.

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
`zero_one_peakfind_density`, `zero_one_gain_guess`, `fit_cols`,
`pedsub_cache_dir`, `force_pedsub`, `use_biweight_loc`, `use_biweight_midvar`,
`plot_zero_one_adu`, `plot_zero_one_electrons`, `electron_fit_mode`, `plot_zero_one_individual`,
`plot_zero_one_together`, `plot_zero_one_yscale`, `plot_zero_one_individual_figsize`,
`plot_zero_one_subplots_figsize`, `plot_zero_one_sharex`, `plot_zero_one_sharey`,
`extra_plot_title`, `nimages`, `show_titles`, `save_plots`, `save_csv`, `show_plots`,
`output_dir`, `verbose`.

`use_overscan_only`, `overscan_cols`, `fit_cols`, and `zero_one_gain_guess` are additions
specific to this package (not present in `nonlinearity_studies`); older configs without
them fall back to the defaults (overscan-only off, all columns used in the fit, gain
auto-detected).

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
