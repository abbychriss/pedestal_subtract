"""
summary — split out of core.py.
"""
import csv
from pathlib import Path

from .constants import _DARK_CURRENT_COLUMN, _DARK_CURRENT_METHODS

def build_extension_summary(gains, double_gauss_popts, dark_current_rows=None,
                            exposure_days=None, raw_pedestals=None):
    """Per-extension rows of gain (ADU/e-) and noise, in both ADU and e-.

    ``noise_adu`` is the zero-electron peak width (``s0``, the first double-Gaussian
    coefficient, in ADU); ``noise_e`` is that same width divided by the gain. When no
    one-electron peak was found the gain is NaN, so noise_e is NaN as well, but
    noise_adu is still reported (it does not depend on the gain).

    When supplied, optional columns are merged in: ``raw_pedestals`` adds
    ``pedestal_raw_adu`` (the pre-subtraction baseline, see ``raw_pedestal_locations``),
    ``exposure_days`` adds an ``exposure_days`` column, and ``dark_current_rows`` (from
    ``calculate_dark_current``) adds the chosen dark-current column(s).
    """
    rows = []
    for ext, gain in enumerate(gains):
        noise_adu = double_gauss_popts[ext][0]  # s0: std of the zero-electron peak in ADU
        row = {'ext': ext + 1}
        if raw_pedestals is not None:
            row['pedestal_raw_adu'] = raw_pedestals[ext]
        row['gain_adu_per_e'] = gain
        row['noise_adu'] = noise_adu
        row['noise_e'] = noise_adu / gain  # ADU -> e- by dividing by gain (ADU/e-)
        if exposure_days is not None:
            row['exposure_days'] = exposure_days
        if dark_current_rows is not None:
            row.update(dark_current_rows[ext])
        rows.append(row)
    return rows


def _extension_summary_fieldnames(dark_current_rows=None, exposure_days=None,
                                  raw_pedestals=None):
    """Ordered CSV field names for the per-extension summary, including the optional
    raw-pedestal, exposure-time and dark-current columns actually present."""
    fieldnames = ['ext']
    if raw_pedestals is not None:
        fieldnames.append('pedestal_raw_adu')
    fieldnames += ['gain_adu_per_e', 'noise_adu', 'noise_e']
    if exposure_days is not None:
        fieldnames.append('exposure_days')
    if dark_current_rows:
        for method in _DARK_CURRENT_METHODS:
            column = _DARK_CURRENT_COLUMN[method]
            if any(column in row for row in dark_current_rows):
                fieldnames.append(column)
    return fieldnames


def write_extension_summary_csv(save_path, gains, double_gauss_popts,
                                dark_current_rows=None, exposure_days=None,
                                raw_pedestals=None):
    """Write the per-extension gain/noise (and optional dark-current) summary as CSV.

    Returns the list of summary rows (see build_extension_summary).
    """
    rows = build_extension_summary(gains, double_gauss_popts, dark_current_rows,
                                   exposure_days, raw_pedestals)
    fieldnames = _extension_summary_fieldnames(dark_current_rows, exposure_days,
                                               raw_pedestals)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return rows
