"""
fits_io — split out of core.py.
"""
import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location
from datetime import datetime
from pathlib import Path

_OVERSCAN_HEADER_KEYS = ('PRESCAN', 'PHYSCOL', 'NCOL', 'NCOLPRE', 'NSBIN')


def get_fits(file_input):
    """
    Load FITS extensions from a file.

    Parameters
    ----------
    file_input : str or Path
        Path to the FITS file (absolute or relative to current working directory)

    Returns
    -------
    ext_charge : list
        List of data arrays from extensions 1–4
    """
    file_path = Path(file_input).resolve()
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"FITS file not found: {file_path}")

    # Load FITS file
    with fits.open(str(file_path)) as hdu_list:
        ext_charge = [hdu_list[i].data for i in range(1, 5)]

    return ext_charge


def get_exposure_time_days(file_input):
    """Exposure time in days, read from the FITS headers as ``DATEEND - DATEINI``.

    Scans every HDU (primary + extensions) for the ISO-8601 ``DATEINI`` (start) and
    ``DATEEND`` (end) timestamps and returns their difference converted to days.

    The exposure time is the per-image integration time even for a stitched frame:
    each pixel of a stitched image was exposed for the same duration as a single
    image (stitching adds pixels, not exposure), and ``stitch_fits`` copies the
    first image's headers, so this returns that single image's exposure as required.

    Returns ``np.nan`` if either timestamp is missing.
    """
    file_path = Path(file_input).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"FITS file not found: {file_path}")

    dateini = dateend = None
    with fits.open(str(file_path)) as hdu_list:
        for hdu in hdu_list:
            if dateini is None and 'DATEINI' in hdu.header:
                dateini = hdu.header['DATEINI']
            if dateend is None and 'DATEEND' in hdu.header:
                dateend = hdu.header['DATEEND']
            if dateini is not None and dateend is not None:
                break

    if dateini is None or dateend is None:
        return np.nan

    t_start = datetime.fromisoformat(str(dateini))
    t_end = datetime.fromisoformat(str(dateend))
    return (t_end - t_start).total_seconds() / 86400.0


def get_pixel_binning(file_input):
    """(NPBIN, NSBIN) = vertical/horizontal pixel binning, from the FITS headers.

    The detector is read out binned, so each image pixel holds the summed charge of
    NPBIN (vertical) x NSBIN (horizontal) physical CCD pixels. Scans every HDU for the
    keys and returns them as ints; a missing key is returned as ``None`` so the caller
    can warn and fall back to 1 (no binning).
    """
    file_path = Path(file_input).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"FITS file not found: {file_path}")

    npbin = nsbin = None
    with fits.open(str(file_path)) as hdu_list:
        for hdu in hdu_list:
            if npbin is None and 'NPBIN' in hdu.header:
                npbin = int(hdu.header['NPBIN'])
            if nsbin is None and 'NSBIN' in hdu.header:
                nsbin = int(hdu.header['NSBIN'])
            if npbin is not None and nsbin is not None:
                break
    return npbin, nsbin


def raw_pedestal_locations(data_ext, use_biweight=False):
    """Robust per-extension pedestal (zero-peak) location of the RAW data.

    Intended to be called on the raw extensions *before* any pedestal subtraction,
    so the reported baseline is the physical pedestal level rather than the ~0
    baseline left after row/column subtraction. Returns one value per extension in
    ADU: the median of the (finite) pixels by default -- robust to the sparse
    positive-charge tail since the zero-electron peak dominates -- or the Tukey
    biweight location when ``use_biweight=True``.
    """
    locations = []
    for data in data_ext:
        flat = np.asarray(data, dtype=float).flatten()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            locations.append(np.nan)
        elif use_biweight:
            locations.append(float(biweight_location(flat)))
        else:
            locations.append(float(np.median(flat)))
    return locations


def get_fits_header(file_input):
    """Return the first FITS header (primary or extension) that carries the CCD
    geometry keys in `_OVERSCAN_HEADER_KEYS`, or the primary header if none do."""
    file_path = Path(file_input).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"FITS file not found: {file_path}")
    with fits.open(str(file_path)) as hdu_list:
        for hdu in hdu_list:
            if all(k in hdu.header for k in _OVERSCAN_HEADER_KEYS):
                return hdu.header
        return hdu_list[0].header


def overscan_cols_from_header(header):
    """Compute the serial-overscan column slice ``(c0, c1)`` from CCD geometry keys.

    The CCD has PRESCAN inactive columns, then PHYSCOL physical columns, then the
    overscan. A frame reads CCD columns ``[NCOLPRE*NSBIN, (NCOL+NCOLPRE)*NSBIN]``,
    so the overscan is every CCD column past ``PRESCAN + PHYSCOL``. Dividing that
    CCD-column count by NSBIN converts it to *binned image* columns, giving the
    last ``n = ((NCOL+NCOLPRE)*NSBIN - (PRESCAN+PHYSCOL)) // NSBIN`` columns of the
    image (for NSBIN=1 this is just the CCD-column count).

    Returns ``(-n, None)`` (a half-open slice selecting the last ``n`` columns),
    or ``None`` if a required key is missing or the computed count is not positive.
    """
    if header is None or not all(k in header for k in _OVERSCAN_HEADER_KEYS):
        return None
    prescan = int(header['PRESCAN'])
    physcol = int(header['PHYSCOL'])
    ncol = int(header['NCOL'])
    ncolpre = int(header['NCOLPRE'])
    nsbin = int(header['NSBIN'])
    if nsbin <= 0:
        return None
    n_overscan_ccd = (ncol + ncolpre) * nsbin - (prescan + physcol)
    n_overscan = n_overscan_ccd // nsbin  # CCD columns -> binned image columns
    if n_overscan <= 0:
        return None
    return (-n_overscan, None)


def _value_for_extension(value, ext, n_ext):
    if isinstance(value, (list, tuple)) and len(value) == n_ext:
        if all(isinstance(v, (list, tuple, np.ndarray)) and len(v) == 2 for v in value):
            return value[ext]
    return value


def _scalar_for_extension(value, ext, n_ext):
    """Select a per-extension scalar from ``value``.

    ``value`` may be a single number/None applied to every extension, or a
    length-``n_ext`` list/tuple of per-extension entries (each a number or None).
    Anything else (e.g. a scalar) is returned unchanged for every extension.
    """
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == n_ext:
        return value[ext]
    return value
