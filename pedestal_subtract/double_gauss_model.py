"""
double_gauss_model — split out of core.py.
"""
import numpy as np

def double_gauss(x, s, m0, m1, N0, N1):
    # Both the zero- and one-electron Gaussians share a single width ``s`` (the readout
    # noise): the peaks are physically the same noise distribution shifted by 1 e-, so
    # one sigma is fit to both. popt layout is (s, m0, m1, N0, N1).
    return N0 * np.exp(-(x-m0)**2/(2*s**2)) + N1 * np.exp(-(x-m1)**2/(2*s**2))
