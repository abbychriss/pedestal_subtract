"""
pedestal_subtract
=================

Standalone pedestal-subtraction + double-Gaussian fitting/plotting pipeline,
extracted from the nonlinearity_studies package.

Pipeline
--------
1. ``get_fits``                    - load extensions 1-4 from a FITS file.
2. ``pedestal_subtract_ext_cached``- row/column pedestal-subtract each extension
                                     (results cached to ``*.pedsub.fits``).
3. ``get_zero_one_peaks_ext``      - fit the zero/one-electron peaks per extension
                                     with a double Gaussian (pedestal, noise, gain).
4. ``plot_zero_one_peaks``         - plot the double-Gaussian fits in ADU and/or
                                     electron units.

Run as a module::

    python -m pedestal_subtract IMAGE.fits

Use ``python -m pedestal_subtract --help`` for all options.
"""

__version__ = "0.1.0"
__author__ = "Abby Chriss"

from .core import (
    convert_to_electrons,
    calculate_noise_gain,
    pedestal_subtract,
    pedestal_subtract_ext_cached,
    get_fits,
    get_zero_one_peaks_ext,
    plot_zero_one_peaks,
    double_gauss,
)

__all__ = [
    "convert_to_electrons",
    "calculate_noise_gain",
    "pedestal_subtract",
    "pedestal_subtract_ext_cached",
    "get_fits",
    "get_zero_one_peaks_ext",
    "plot_zero_one_peaks",
    "double_gauss",
]
