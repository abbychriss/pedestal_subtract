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
   `*.pedsub.fits` file in a `cache/` folder beside the source so reruns with the
   same parameters skip the recompute (`pedestal_subtract_ext_cached`).
3. **Fits the zero/one-electron peaks** of each extension with a double Gaussian whose
   two peaks share a single width (the read noise), yielding the pedestal, read noise,
   and gain (`get_zero_one_peaks_ext`).
4. **Plots** those double-Gaussian fits in ADU and/or electron units, individually
   per extension and/or as a combined 2×2 subplot (`plot_zero_one_peaks`).

## Code structure

The package is organized by responsibility rather than as one large `core` module:

- `double_gauss_model.py` — the double-Gaussian model function
- `calibrate.py` — ADU→electron conversion and noise/gain calculation
- `fit_zero_one.py` — histogramming and zero/one-electron peak fitting
- `pedestal.py` — pedestal subtraction, overscan handling, and caching
- `fits_io.py` — FITS loading and header/geometry readers
- `dark_current.py` — dark-current estimation methods
- `plotting.py` — zero/one, dark-current, and charge-per-column figures
- `summary.py` — per-extension summary tables (CSV)
- `__main__.py` — command-line entry point; `cli_config.py` holds its config/argument helpers
- `core.py` — backwards-compatibility shim that re-exports the full public API, so existing `from pedestal_subtract.core import ...` imports keep working

FITS stitching lives in the shared `analysis_tools` package.

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
    --save_output --no-show_plots --plot_zero_one_electrons -o ./plots
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
| `--zero_one_n_bins` | `400` | The single zero/one binning knob: a strict number of bins used for **both** the internal peak-finding histograms and the double-Gaussian fit/plot histogram. Every window (test range, peak-search range, fit window) is divided into this many bins, so widening the window (via the scales) coarsens the bin width instead of adding bins. Raise for finer bins; lower to aggregate sparse hits in low-statistics images. Clamped internally to [10, 4000]. Integer ≥ 1. |
| `--zero_one_window_left_scale` | `1.0` | Scale the left half-width of the auto-computed zero/one fit window (≥ 1.0; > 1 widens toward lower charge). Widening coarsens the bin width (see `--zero_one_n_bins`). |
| `--zero_one_window_right_scale` | `1.0` | Scale the right half-width of the auto-computed zero/one fit window (≥ 1.0; > 1 widens past the one-electron peak). Widening coarsens the bin width (see `--zero_one_n_bins`). |
| `--fit_cols SPEC ...` | all columns | Restrict the zero/one fit (and the plotted histograms) to image columns. A region is a Python half-open slice `START:STOP` (either end may be blank — `3400:` runs to the end, `:3200` from the start; negative endpoints count from the right). Join several regions with commas to keep their **union**, e.g. `0:3200,3400:3500`. Pass `auto` to keep every column **except** the hot ones (median charge ≥ `--n_std_to_mask` biweight-SDs from the pedestal location — the same columns drawn red by `--plot_charge_per_column`); a manual region entry overrides `auto`. Give one value to apply it to **every** extension, or one **per** extension (use `null` for an extension to keep all its columns). Bare integers still work for a single range: two ints (`START STOP`) for all extensions, or two per extension (e.g. 8 ints for 4 extensions). In the JSON config use `"auto"`, a `"lo:hi,lo:hi"` string, a `[START, STOP]` pair, or a per-extension list of those/`null` (e.g. `["0:3200,3400:3500", null, "auto", [10, -10]]`). Applied **after** pedestal subtraction, so the per-row pedestal / overscan estimate still uses the full frame. Omit to use every column. |
| `--zero_one_gain_guess GAIN ...` | auto-detect | Seed for the one-electron peak location — a guess for the gain (ADU/e⁻). The seed places `μ₁` at `μ₀ + gain` and **restricts the fit bound on `μ₁` to a tight ±0.05 ADU band** around that location, so the fit stays on the guessed peak instead of sliding down onto the zero-peak tail. It does **not** affect the fit window size or the binning (those come from the auto-detected bump and `--zero_one_n_bins`). The post-fit acceptance band (gain ∈ [0.5, 1.8] ADU/e⁻) is unchanged. Pass one value to apply it to **every** extension, or one **per** extension (e.g. 4 values for 4 extensions). In the JSON config use a single number or a per-extension list of numbers/`null` (e.g. `[1.05, null, 0.95, 1.1]`, where `null` auto-detects that extension). Values must be > 0. Omit to auto-detect for all. |
| `--force_pedsub` | off | Recompute, ignoring the on-disk cache. |
| `--pedsub_cache_dir DIR` | `cache/` beside source | Where to write the `*.pedsub.fits` cache. |
| `-z` / `--plot_zero_one_adu` | on | Plot fits in ADU. |
| `--plot_zero_one_electrons` | off | Also plot in electron units. |
| `--plot_dark_current` | on | Plot the electron-unit zero/one distributions with a legend of σ, gain and the dark current(s). When the `count` method is used, the count window is marked by vertical dashed lines (±`dark_current_count_nsigma`·σ about 1 e⁻); when it is not, those lines are omitted and (if the `weighted` method is used) the legend instead reports the weighted dark current and its formula `n_events = N₁/(N₁+N₀)`. |
| `--plot_charge_per_column_together` | off | Plot the median charge per column for each extension (2×2 grid) on the **raw**, pre-pedestal-subtraction data — a diagnostic for anomalous columns. A column whose median sits ≥ `--n_std_to_mask` biweight-SDs from the extension's biweight location is flagged "hot" (drawn red). Every column is plotted; the columns **excluded** by `--fit_cols` are shaded light grey so the masked-out region is visible (with `--fit_cols auto`, the shaded columns are exactly the red hot ones). With `--verbose`, the hot columns per extension are also printed. |
| `--plot_charge_per_column_individual` | off | Same as `--plot_charge_per_column_together`, but draws one figure per extension instead of a combined 2×2 grid. |
| `--electron_fit_mode` | `transform` | How the electron-unit curve is obtained. `transform` analytically rescales the converged ADU fit (exact; μ₀ = 0 / μ₁ = 1 by construction, no refit). `refit` fits the double Gaussian again directly to the electron-unit histogram, letting the peaks/widths re-optimise in electron space (widths kept positive, amplitudes non-negative, means free; falls back to the transform if the refit fails). Only affects the electron-units plot. |
| `--plot_zero_one_yscale` | `linear` | `linear` or `log`. |
| `-s` / `--save_output` | off | Save plots to `--output_dir` **and** write a per-extension `extension_summary.csv` (columns `ext`, `pedestal_raw_adu`, `gain_adu_per_e`, `noise_adu`, `noise_e`, `exposure_days`, and the chosen dark-current column(s)) into the run's output folder. `pedestal_raw_adu` is the median pedestal location of the **raw** data, before pedestal subtraction. `noise_adu` is the shared peak width (σ) in ADU; `noise_e` is that width divided by the gain. For an extension with no one-electron peak, `gain_adu_per_e` and `noise_e` are `nan`, but `noise_adu` is still reported (it does not depend on the gain). |
| `--do_dark_current` | on | Compute the per-extension dark current. Use `--no-do_dark_current` (or `"do_dark_current": false` in the config) to skip the dark-current calculation and its plot entirely; `extension_summary.csv` then omits the dark-current column(s). |
| `--dark_current_method` | `weighted` | How to count single-electron events for the **dark current** (electrons / physical pixel / day = single-electron events ÷ physical pixels ÷ exposure-days, where physical pixels = image pixels × NPBIN × NSBIN, since the detector is read out binned and each image pixel sums NPBIN×NSBIN physical pixels). `count` counts pixels within `dark_current_count_nsigma` peak widths σ of the 1 e⁻ charge, then divides by the Gaussian fraction `erf(nsigma ⁄ √2)` (0.6827 at ±1σ) to estimate the **full** one-electron count, bringing it into agreement with the other two methods (the single shared σ, the pedestal width, is well-determined even at low statistics; the zero-peak tail is **not** subtracted, since those pixels can't be physically distinguished from real 1 e⁻ events). `integrate` uses the analytic area under the fitted one-electron Gaussian, `N₁·σ·√(2π) ⁄ bin-width` (a conservative upper bound, sensitive to the fit). `weighted` uses the one-electron amplitude fraction `n_events = N₁ ⁄ (N₁ + N₀)`, which is already per image pixel; its rate is therefore `N₁ ⁄ (N₁ + N₀) ⁄ (NPBIN·NSBIN) ⁄ exposure-days` — divided only by the pixel binning and exposure, **not** by the pixel count (which cancels in the ratio). `all` computes each and writes a column per method (`dark_current_count_e_per_pix_day`, `dark_current_integrate_e_per_pix_day`, `dark_current_weighted_e_per_pix_day`) so they can be compared. A `null` config value (or omitting the key) selects the default method, `weighted`. The exposure time is `DATEEND − DATEINI` from the FITS header (or `--exposure_time_s` when set) — unchanged for a stitched image, since each pixel was exposed for one image's duration. When the gain is undefined or the exposure can't be read, the dark current is `nan`. |
| `--dark_current_count_center` | `one_electron` | For the `count` method, where the window is centred: `one_electron` (default) on the ideal 1 e⁻ charge (`pedestal + gain`), or `mu1` on the fitted one-electron peak mean μ₁. |
| `--dark_current_count_nsigma` | `1.0` | Half-width of the `count` window in units of the fitted peak width σ (default `1.0` = ±1σ). The single shared σ (the pedestal width) stays well-determined at low statistics. Use `< 1` to shrink the window and cut the zero-peak tail contamination that dominates the count at low dark current. The rate itself does **not** depend on how many rows/columns are analysed (event count and pixel count scale together), so the charge-window width — not the spatial extent — is the knob for tail contamination. |
| `--exposure_time_s` | from header | Manually set the exposure time in **seconds**, overriding the FITS-header `DATEEND − DATEINI`. Use when a file's header is missing `DATEINI`/`DATEEND` (so the dark current would otherwise be `nan`). Converted to days internally. In the JSON config use `null` to read from the header (the default). |
| `--no-show_plots` | (shows) | Don't open figures interactively. |

When saving to disk, each run is stored in a unique subfolder named with a short
hash of the analysis-relevant config. The complete resolved config is also
written to `config.json` inside that folder for reproducibility.

### Low-statistics one-electron peaks

The double-Gaussian fit is designed to behave sensibly when an extension has only
a handful of single-electron hits:

- **The fit is Poisson-weighted** (each bin weighted by `1/√counts`), so the
  ~10-count one-electron region genuinely constrains the fit instead of being
  drowned out by the ~10⁴-count zero peak. Without the weighting, the fit could
  "cheaply" widen the zero Gaussian and shift its mean slightly to absorb the
  one-electron shoulder, parking the second Gaussian anywhere.
- **The far left tail is excluded from the fit.** The zero peak can carry a
  heavy non-Gaussian left tail (a pedestal-subtraction artefact); fed to a
  symmetric Gaussian it inflates σ and pulls μ₀ below zero. The fit therefore
  only uses bins from `median − 2·σ_R` upward, where `σ_R` is the robust
  right-half sigma of the data (`_ZERO_PEAK_LEFT_FIT_NSIGMA` in
  `constants.py`). The histogram/plot window is unaffected, so an anomalous
  left tail remains visible against the fitted curve.
- **The gain (ADU/e⁻) is bounded**: the fitted peak separation must land in
  `[0.5, 1.8]` ADU/e⁻ or the one-electron peak is rejected as not a real 0→1
  step. The band edges are the `_MIN_GAIN_ADU` / `_MAX_GAIN_ADU` constants in
  `constants.py`.
- **μ₁ is kept off the zero-peak shoulder, but never pushed past the real peak.**
  The one-electron mean is bounded below by ~4 zero-peak widths off the pedestal
  (so it cannot slide down onto the bright shoulder), and that guard is capped at
  ~one width below the detected bump (so a **wide** zero peak — where 4 widths
  exceed the actual gain — cannot exclude the genuine peak from the fit). The
  bump is seeded from the largest excess above the estimated zero-peak tail,
  searched only past the shoulder; the zero-peak width itself is capped by the
  robust MAD sigma of the windowed data so coarse binning cannot inflate it. (A
  stricter "localized excess, not a tail" post-fit check,
  `_one_electron_peak_is_real`, exists but is currently disabled.)
- **N₁ is tied to the measured bump height.** The one-electron amplitude is
  bounded to the largest fit-histogram count in the allowed μ₁ band — after
  subtracting the zero Gaussian's estimated contribution per bin — within an
  asymmetric Poisson band: at most 1√count below but up to 2√count above the
  measured height (the `1/√counts` weighting systematically biases a sparse
  peak's amplitude low, so the tight side is below). The fit therefore cannot
  deflate N₁ while the zero Gaussian soaks up the bump; an empty band leaves
  the lower bound at 0, so a genuinely absent peak is still reported as "no
  peak".

When no genuine peak is found, the gain is reported as undefined (`NaN`), the
console prints `no 1 e- peak found`, and the plot shows only the single
zero-electron Gaussian. The histogram bin count (for both peak detection and the
fit/plot) is set by the single knob `--zero_one_n_bins`.

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
`zero_one_gain_guess`, `fit_cols`,
`pedsub_cache_dir`, `force_pedsub`, `use_biweight_loc`, `use_biweight_midvar`,
`plot_zero_one_adu`, `plot_zero_one_electrons`, `plot_dark_current`,
`plot_charge_per_column_together`, `plot_charge_per_column_individual`, `electron_fit_mode`, `plot_zero_one_individual`,
`plot_zero_one_together`, `plot_zero_one_yscale`, `plot_zero_one_individual_figsize`,
`plot_zero_one_subplots_figsize`, `plot_dark_current_figsize`, `plot_charge_per_column_figsize`,
`plot_charge_per_column_individual_figsize`,
`plot_zero_one_sharex`, `plot_zero_one_sharey`,
`extra_plot_title`, `nimages`, `show_titles`, `save_output`,
`do_dark_current`, `dark_current_method`, `dark_current_count_center`, `dark_current_count_nsigma`,
`exposure_time_s`, `show_plots`, `output_dir`, `verbose`.

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
