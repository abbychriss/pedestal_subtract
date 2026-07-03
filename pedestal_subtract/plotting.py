"""
plotting — split out of core.py.
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import biweight_location
from astropy.stats import biweight_midvariance
from pathlib import Path
from scipy.optimize import curve_fit

from .calibrate import convert_to_electrons
from .double_gauss_model import double_gauss

_SUBPLOTS_FIGSIZE = (13, 9)


_INDIVIDUAL_FIGSIZE = (8, 6)


def _bar_heights(counts, yscale):
    """Bar heights for plotting.

    On a log y-axis, zero-count bins are set to NaN so matplotlib draws nothing
    there -- the bins are genuinely empty rather than clamped up to 1, which
    previously produced a solid rectangular block along the bottom of images
    dominated by empty (zero-electron) bins.
    """
    counts = np.asarray(counts, dtype=float)
    if yscale == 'log':
        heights = counts.copy()
        heights[heights <= 0] = np.nan
        return heights
    return counts


def _fit_curve_x(lo, hi, n_bins, points_per_bin=20, floor=500):
    """x-grid for drawing the fitted curve, sampled at `points_per_bin` points per
    histogram bin so the curve stays smooth when zoomed in -- the point density
    per unit charge is constant regardless of how wide the window is scaled."""
    n = max(floor, int(points_per_bin) * int(n_bins))
    return np.linspace(lo, hi, n)


def _finish_fig(show_plots):
    """Display the current figure interactively (blocking), or close it when not showing.

    plt.show() displays every figure still registered with pyplot, so the current
    figure must be closed once the user dismisses its window; otherwise the next
    figure's plt.show() re-raises this one too and interactive GUI backends (notably
    macOS) pop up duplicates. Closing right after the blocking show keeps figures
    appearing one at a time with a clean registry. When not showing, the figure is
    closed immediately to free memory.
    """
    if show_plots:
        plt.show()
    plt.close()


def _zero_one_ylim(ylim, yscale, counts, fit_y):
    """Per-extension (low, high) for a zero-one subplot, or None to leave autoscaling.

    `counts`/`fit_y` are the bar heights and fitted curve for the extension and drive the
    autoscale ('none'/None) case, so callers can share a y-axis across extensions by taking
    the minimum low and maximum high of every extension's returned range.
    """
    if isinstance(ylim, str) and ylim == 'default':
        if yscale == 'log':
            return 0.5, np.max(counts) * 1e4
        if yscale == 'linear':
            return 0, np.max(counts) + 2.5e4
        return None
    if ylim is None or ylim == 'none':
        if yscale != 'linear':
            return None  # let matplotlib pick a sensible (e.g. log) range per axis
        return 0, max(np.max(counts), np.max(fit_y)) * 1.05
    return ylim[0], ylim[1]


def _double_gauss_popt_electrons(double_gauss_popt, pedestal, gain):
    """Transform the converged ADU double-Gaussian fit into electron units.

    Converting charge to electrons, ``x_e = (x - pedestal) / gain``, is an exact
    linear rescaling under which the double Gaussian maps term for term: the means
    go to ``mu0_e = 0`` and ``mu1_e = 1`` (by construction, since ``pedestal == mu0``
    and ``gain == mu1 - mu0``), the widths scale as ``sigma / gain``, and the
    amplitudes are unchanged (the electron histogram uses the same bin count over
    the rescaled range, so every bin holds the same pixels). The electron-unit curve
    is therefore fully determined by the ADU fit.

    We transform analytically rather than re-fitting in electron units: an
    independent refit has nothing to gain (the answer is fixed by the transform) and
    its free parameters only let curve_fit drift the one-electron peak off
    ``mu_1 = 1`` and balloon the shared ``sigma`` -- exactly the spurious wide/displaced
    electron-space fit otherwise seen even when the ADU fit is clean.
    """
    s, m0, m1, N0, N1 = (double_gauss_popt[0], double_gauss_popt[1], double_gauss_popt[2],
                         double_gauss_popt[3], double_gauss_popt[4])
    return np.array([s / gain, (m0 - pedestal) / gain, (m1 - pedestal) / gain, N0, N1])


def _electron_double_gauss_popt(double_gauss_popt, pedestal, gain, centers_e, counts_e,
                                electron_fit_mode='transform'):
    """Double-Gaussian parameters in electron units, by one of two methods.

    ``'transform'`` (default): analytically rescale the converged ADU fit
    (``_double_gauss_popt_electrons`` -- exact, with ``mu_0 = 0`` / ``mu_1 = 1`` fixed
    by construction and no refit).

    ``'refit'``: fit the double Gaussian directly to the electron-unit histogram
    (``centers_e``/``counts_e``), starting from that analytic transform, so the
    electron-space peaks and widths are re-optimised rather than pinned by the ADU
    fit. Widths are kept positive and amplitudes non-negative; the means are left
    free, so the one-electron peak need not land exactly at 1 e-. Falls back to the
    analytic transform if the refit fails to converge. (See the note in
    ``_double_gauss_popt_electrons`` on why the transform is the default -- a refit
    can drift ``mu_1`` off 1 and balloon the shared ``sigma`` even on a clean ADU fit.)
    """
    transform = _double_gauss_popt_electrons(double_gauss_popt, pedestal, gain)
    if electron_fit_mode != 'refit':
        return transform
    lo = [1e-12, -np.inf, -np.inf, 0.0, 0.0]  # (s, m0, m1, N0, N1)
    hi = [np.inf] * 5
    try:
        popt_e, _ = curve_fit(double_gauss, centers_e, counts_e, p0=transform,
                              maxfev=20000, bounds=(lo, hi))
        return popt_e
    except (RuntimeError, ValueError):
        return transform


def _zero_one_label_adu(double_gauss_popt, gain):
    """Legend label for an ADU zero-one subplot.

    When `gain` is NaN no trustworthy one-electron peak was found, so only the
    zero-peak parameters are shown (the one-electron amplitude has been zeroed,
    making the plotted curve a single Gaussian) and the gain is reported as
    undefined rather than printing a meaningless value.
    """
    s, m0, m1, N0, N1 = (double_gauss_popt[0], double_gauss_popt[1], double_gauss_popt[2],
                         double_gauss_popt[3], double_gauss_popt[4])
    if np.isnan(gain):
        return (r'$\sigma$ = %5.3f, $\mu_0$ = %5.3f, $N_0$ = %5.3f' % (s, m0, N0)
                + '\n' + r'no $1\,e^{–}$ peak found (gain undefined)')
    return (r'$\sigma$ = %5.3f, $\mu_0$ = %5.3f, $\mu_1$ = %5.3f,' % (s, m0, m1)
            + '\n' + r'$N_0$ = %5.3f, $N_1$ = %5.3f, gain = %5.3f ADU/$e^{–}$' % (N0, N1, gain))


def plot_zero_one_peaks(data_ext,
                        zero_one_counts_ext,
                        zero_one_edges_ext,
                        pedestals, 
                        gains, 
                        double_gauss_popts, 
                        zero_one_ranges,
                        individual_figsize=_INDIVIDUAL_FIGSIZE,
                        subplots_figsize=_SUBPLOTS_FIGSIZE,
                        xlim='default',
                        ylim='default',
                        ylim_electrons='default',
                        suptitle='Double-Gaussian Fit to Zero-One Electron Peaks',
                        additional_title='',
                        nimages=10,
                        fontsize=9.5,
                        yscale='linear',
                        n=100,
                        do_convert_to_electrons=False,
                        electron_fit_mode='transform',
                        plot_individual=False,
                        plot_together=True,
                        do_plot_adu=True,
                        sharex=False,
                        sharey=False,
                        show_titles=True,
                        save_plots=False,
                        show_plots=True,
                        fig_path='./', file='zero_one_peaks',
                        dpi=350):

    fig_path = Path(fig_path)
    # Build the output base from the source filename with its final extension stripped.
    # `base_name` is treated as a plain string and joined with the variant suffix and
    # '.jpeg' directly (see the save sites below) -- using Path.with_suffix/.with_stem
    # here would mistake any '.' in the stem (e.g. a voltage like VR-8.5) for the file
    # extension and truncate the name there.
    if file != 'zero_one_peaks':
        base_name = Path(file).stem + '_zero_one_peaks'
    else:
        base_name = file

    if plot_individual:
        for ext, data in (enumerate(data_ext) if do_plot_adu else []):
            data = np.array(data).flatten()

            zero_one_counts=zero_one_counts_ext[ext]
            zero_one_edges=zero_one_edges_ext[ext]
            pedestal=pedestals[ext]
            gain=gains[ext]
            double_gauss_popt=double_gauss_popts[ext]
            zero_one_range=zero_one_ranges[ext]

            fig, ax = plt.subplots(1, 1, figsize=individual_figsize, constrained_layout=True)
            if show_titles:
                ax.set_title(f'{additional_title}{suptitle} (Nimages = {nimages}): EXT {ext + 1}', fontsize=12, pad=10)
            ax.set_xlabel('Charge (ADU)')
            ax.set_ylabel('N')

            window_max = np.max(zero_one_counts)  # zero-peak height, drives y-scaling

            if yscale=='log':
                ax.set_yscale('log')
            elif yscale!='linear':
                ax.set_yscale(yscale)
            ax.stairs(_bar_heights(zero_one_counts, yscale), zero_one_edges, fill=True)

            curve_x = _fit_curve_x(zero_one_range[0], zero_one_range[1], len(zero_one_counts))
            ax.plot(curve_x, double_gauss(curve_x, *double_gauss_popt), 'r',
                label=_zero_one_label_adu(double_gauss_popt, gain))
            ax.legend(loc="upper right", fontsize=fontsize)

            if xlim=='default':
                ax.set_xlim(zero_one_range[0], zero_one_range[1])
            elif xlim is not None and xlim!='none':
                ax.set_xlim(xlim)
            # xlim is None or 'none': leave autoscaled to the full data extent

            if ylim=='default':
                if yscale=='log':
                    ax.set_ylim(0.5, window_max * 1e4)
                elif yscale=='linear':
                    ax.set_ylim(0, window_max + 3e4)
            elif ylim!='none':
                ax.set_ylim(ylim)

            if save_plots:
                output_path = fig_path / f'{base_name}_EXT{ext+1}.jpeg'
                plt.savefig(str(output_path), dpi=dpi)
                print(f'Saved plots to {output_path}')
            _finish_fig(show_plots)

        if do_convert_to_electrons:
            for ext, data in enumerate(data_ext):
                data=np.array(data).flatten()
                zero_one_range = zero_one_ranges[ext]
                pedestal = pedestals[ext]
                gain = gains[ext]

                # No one-electron peak -> no gain -> electron conversion (divide by
                # gain) is undefined; skip the electron plot for this extension.
                if np.isnan(gain):
                    continue

                data_window=data[(data > zero_one_range[0]) & (data < zero_one_range[1])]

                data_window_e = convert_to_electrons(data_window, pedestal, gain)
                zero_one_range_e = convert_to_electrons(zero_one_range, pedestal, gain)
                # Match the (scale-adjusted) ADU fit-window bin count so the electron
                # plot keeps the same bins-to-window ratio when the window is widened.
                nbins = len(zero_one_counts_ext[ext])

                # Window histogram (drawn as the bars).
                zero_one_counts_e, zero_one_edges_e = np.histogram(data_window_e, bins=nbins, range=zero_one_range_e)
                zero_one_centers_e = 0.5 * (zero_one_edges_e[:-1] + zero_one_edges_e[1:])
                window_max_e = np.max(zero_one_counts_e)

                # Electron-unit curve: the ADU fit rescaled exactly, or a direct refit in
                # electron units (see electron_fit_mode / _electron_double_gauss_popt).
                double_gauss_popt_e = _electron_double_gauss_popt(
                    double_gauss_popts[ext], pedestal, gain,
                    zero_one_centers_e, zero_one_counts_e, electron_fit_mode)

                fig, ax = plt.subplots(1, 1, figsize=individual_figsize, constrained_layout=True)
                if show_titles:
                    ax.set_title(rf'{additional_title}{suptitle} (Nimages = {nimages}): EXT {ext + 1}', fontsize=12, pad=10)

                if yscale=='log':
                    ax.set_yscale('log')
                elif yscale!='linear':
                    ax.set_yscale(yscale)

                ax.stairs(_bar_heights(zero_one_counts_e, yscale), zero_one_edges_e, fill=True)
                ax.set_xlabel(r'Charge ($e^–$)')
                ax.set_ylabel('N')
                curve_x_e = _fit_curve_x(zero_one_range_e[0], zero_one_range_e[1], len(zero_one_counts_e))
                ax.plot(curve_x_e, double_gauss(curve_x_e, *double_gauss_popt_e), 'r',
                    label=r'$\sigma$ = %5.3f $e^{–}$, $\mu_0$ = %5.3f $e^{–}$, $\mu_1$ = %5.3f $e^{–}$,'%tuple(double_gauss_popt_e)[0:3]
                    +'\n'+'gain = %5.3f ADU/$e^{–}$'%gain)
                ax.legend(loc="upper right", fontsize=fontsize)

                if xlim=='default':
                    ax.set_xlim(zero_one_range_e[0], zero_one_range_e[1])
                elif xlim is not None and xlim!='none':
                    ax.set_xlim(xlim)
                # xlim is None or 'none': leave autoscaled to the full data extent

                if ylim_electrons=='default':
                    if yscale=='log':
                        ax.set_ylim(0.5, window_max_e * 1e4)
                    elif yscale=='linear':
                        ax.set_ylim(0, window_max_e + 2.5e4)
                elif ylim_electrons!='none':
                    ax.set_ylim(ylim_electrons)

                if save_plots:
                    output_path = fig_path / f'{base_name}_electrons_EXT{ext+1}.jpeg'
                    plt.savefig(str(output_path), dpi=dpi)
                    print(f'Saved plot to {output_path}')
                _finish_fig(show_plots)

    if plot_together:

        if do_plot_adu:
            fig, axs = plt.subplots(2, 2, figsize=subplots_figsize, constrained_layout=True, sharex=sharex, sharey=sharey)
            if show_titles:
                fig.suptitle(f'{additional_title}{suptitle} (Nimages = {nimages})')
            axs = axs.flatten()

            _ylims = []
            for ext, data in enumerate(data_ext):
                data = np.array(data).flatten()
                zero_one_counts=zero_one_counts_ext[ext]
                zero_one_edges=zero_one_edges_ext[ext]
                pedestal=pedestals[ext]
                gain=gains[ext]
                double_gauss_popt=double_gauss_popts[ext]
                zero_one_range=zero_one_ranges[ext]

                ax = axs[ext]

                if yscale=='log':
                    ax.set_yscale('log')
                elif yscale!='linear':
                    ax.set_yscale(yscale)

                ax.stairs(_bar_heights(zero_one_counts, yscale), zero_one_edges, fill=True)

                curve_x = _fit_curve_x(zero_one_range[0], zero_one_range[1], len(zero_one_counts))
                fit_y = double_gauss(curve_x, *double_gauss_popt)
                ax.plot(curve_x, fit_y, 'r',
                    label=_zero_one_label_adu(double_gauss_popt, gain))

                ax.set_xlabel('Charge (ADU)')
                ax.set_ylabel('N')
                ax.set_title(f'EXT {ext + 1}')
                ax.legend(loc="upper right", fontsize=fontsize - 2)

                if xlim=='default':
                    ax.set_xlim(zero_one_range[0], zero_one_range[1])
                elif xlim is not None and xlim!='none':
                    ax.set_xlim(xlim)
                # xlim is None or 'none': leave autoscaled to the full data extent

                lim = _zero_one_ylim(ylim, yscale, zero_one_counts, fit_y)
                _ylims.append(lim)
                if not sharey and lim is not None:
                    ax.set_ylim(lim)

            # With sharey=True every subplot shares one range; span every extension by
            # taking the minimum low and maximum high so no peak is clipped (the tallest
            # peak need not be in the same extension as the lowest floor).
            _valid = [lim for lim in _ylims if lim is not None]
            if sharey and _valid:
                axs[0].set_ylim(min(lo for lo, _ in _valid), max(hi for _, hi in _valid))

            for i in (0, 1):
                axs[i].set_xlabel('')
                axs[i].tick_params(labelbottom=True)
            for i in (1, 3):
                axs[i].set_ylabel('')
                axs[i].tick_params(labelleft=True)

            if save_plots:
                output_path = fig_path / f'{base_name}.jpeg'
                plt.savefig(str(output_path), dpi=dpi)
                print(f'Saved plot to {output_path}')
            _finish_fig(show_plots)

        if do_convert_to_electrons:
            fig, axs = plt.subplots(2, 2, figsize=subplots_figsize, constrained_layout=True, sharex=sharex, sharey=sharey)
            if show_titles:
                fig.suptitle(rf'{additional_title}{suptitle} (Nimages = {nimages})')
            axs = axs.flatten()

            _ylims = []
            for ext, data in enumerate(data_ext):
                ax = axs[ext]

                data = np.array(data).flatten()
                zero_one_range = zero_one_ranges[ext]
                pedestal = pedestals[ext]
                gain = gains[ext]

                # No one-electron peak -> gain undefined -> cannot convert to
                # electrons; leave the subplot empty with a note and move on.
                if np.isnan(gain):
                    ax.set_title(f'EXT {ext + 1}')
                    ax.set_xlabel(r'Charge ($e^–$)')
                    ax.set_ylabel('N')
                    ax.text(0.5, 0.5, r'no $1\,e^{–}$ peak found',
                            ha='center', va='center', transform=ax.transAxes)
                    _ylims.append(None)
                    continue

                data_window = data[(data > zero_one_range[0]) & (data < zero_one_range[1])]

                data_window_e = convert_to_electrons(data_window, pedestal, gain)
                zero_one_range_e = convert_to_electrons(zero_one_range, pedestal, gain)
                # Match the (scale-adjusted) ADU fit-window bin count so the electron
                # plot keeps the same bins-to-window ratio when the window is widened.
                nbins = len(zero_one_counts_ext[ext])

                # Window histogram (drawn as the bars).
                zero_one_counts_e, zero_one_edges_e = np.histogram(data_window_e, bins=nbins, range=zero_one_range_e)
                zero_one_centers_e = 0.5 * (zero_one_edges_e[:-1] + zero_one_edges_e[1:])

                # Electron-unit curve: the ADU fit rescaled exactly, or a direct refit in
                # electron units (see electron_fit_mode / _electron_double_gauss_popt).
                double_gauss_popt_e = _electron_double_gauss_popt(
                    double_gauss_popts[ext], pedestal, gain,
                    zero_one_centers_e, zero_one_counts_e, electron_fit_mode)

                if yscale=='log':
                    ax.set_yscale('log')
                elif yscale!='linear':
                    ax.set_yscale(yscale)

                ax.stairs(_bar_heights(zero_one_counts_e, yscale), zero_one_edges_e, fill=True)

                ax.set_title(f'EXT {ext + 1}')
                curve_x_e = _fit_curve_x(zero_one_range_e[0], zero_one_range_e[1], len(zero_one_counts_e))
                fit_y_e = double_gauss(curve_x_e, *double_gauss_popt_e)
                ax.plot(curve_x_e, fit_y_e, 'r',
                    label=r'$\sigma$ = %5.3f $e^{–}$, $\mu_0$ = %5.3f $e^{–}$, $\mu_1$ = %5.3f $e^{–}$,'%tuple(double_gauss_popt_e)[0:3]
                    +'\n'+'gain = %5.3f ADU/$e^{–}$'%gain)
                ax.legend(loc="upper right", fontsize=fontsize - 2)
                ax.set_xlabel(r'Charge ($e^–$)')
                ax.set_ylabel('N')

                if xlim=='default':
                    ax.set_xlim(zero_one_range_e[0], zero_one_range_e[1])
                elif xlim is not None and xlim!='none':
                    ax.set_xlim(xlim)
                # xlim is None or 'none': leave autoscaled to the full data extent

                lim = _zero_one_ylim(ylim_electrons, yscale, zero_one_counts_e, fit_y_e)
                _ylims.append(lim)
                if not sharey and lim is not None:
                    ax.set_ylim(lim)

            _valid = [lim for lim in _ylims if lim is not None]
            if sharey and _valid:
                axs[0].set_ylim(min(lo for lo, _ in _valid), max(hi for _, hi in _valid))

            for i in (0, 1):
                axs[i].set_xlabel('')
                axs[i].tick_params(labelbottom=True)
            for i in (1, 3):
                axs[i].set_ylabel('')
                axs[i].tick_params(labelleft=True)

            if save_plots:
                output_path = fig_path / f'{base_name}_electrons.jpeg'
                plt.savefig(str(output_path), dpi=dpi)
                print(f'Saved plot to {output_path}')
            _finish_fig(show_plots)


def _dark_current_fit_label(sigma_e, gain, N1):
    """Legend text for the fit curve: shared width (e-), gain, and 1 e- amplitude N1."""
    return (r'$\sigma$ = %.3f $e^-$' % sigma_e
            + '\n' + r'gain = %.3f ADU/$e^-$, $N_1$ = %.4g' % (gain, N1))


def plot_dark_current_zero_one(data_ext, zero_one_counts_ext, zero_one_edges_ext,
                               pedestals, gains, double_gauss_popts, zero_one_ranges,
                               dark_current_rows, count_center='one_electron',
                               count_nsigma=1.0, electron_fit_mode='transform',
                               nimages=10, figsize=_SUBPLOTS_FIGSIZE, fontsize=10, yscale='log',
                               suptitle='Dark-Current Window (Zero-One Peaks, electrons)',
                               additional_title='', show_titles=True,
                               save_plots=False, show_plots=True,
                               fig_path='./', file='dark_current', dpi=350):
    """Plot the electron-unit zero/one distributions with the dark-current count window.

    One subplot per extension (2x2): the electron-unit histogram and fitted
    double-Gaussian curve, with one independent legend entry per dark-current method
    actually present in ``dark_current_rows`` for that extension (the shared-sigma/
    gain/N1 fit-curve entry is always shown in addition). When the 'count' method was
    used, vertical dashed lines mark the +/- ``count_nsigma`` * sigma charge window it
    integrates around 1 e-. When the 'integrate' method was used, the area under the
    fitted one-electron Gaussian component is shaded. When the 'weighted' method was
    used, its dark current and formula are reported as a legend-only entry. Any
    combination of these may appear together (up to 4 legend entries total).
    Extensions with no one-electron peak (gain undefined) cannot be converted to
    electrons and are left with an explanatory note.
    """
    fig_path = Path(fig_path)
    base_name = Path(file).stem + '_dark_current' if file != 'dark_current' else file

    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    if show_titles:
        fig.suptitle(f'{additional_title}{suptitle} (Nimages = {nimages})')
    axs = axs.flatten()

    for ext, data in enumerate(data_ext):
        ax = axs[ext]
        ax.set_title(f'EXT {ext + 1}')
        ax.set_xlabel(r'Charge ($e^-$)')
        ax.set_ylabel('N')

        pedestal = pedestals[ext]
        gain = gains[ext]
        if np.isnan(gain):
            ax.text(0.5, 0.5, r'no $1\,e^-$ peak found', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        data = np.asarray(data).flatten()
        zero_one_range = zero_one_ranges[ext]
        data_window = data[(data > zero_one_range[0]) & (data < zero_one_range[1])]
        data_window_e = convert_to_electrons(data_window, pedestal, gain)
        zero_one_range_e = convert_to_electrons(np.asarray(zero_one_range), pedestal, gain,
                                                flatten=False)
        nbins = len(zero_one_counts_ext[ext])
        counts_e, edges_e = np.histogram(data_window_e, bins=nbins, range=zero_one_range_e)
        centers_e = 0.5 * (edges_e[:-1] + edges_e[1:])

        popt_e = _electron_double_gauss_popt(double_gauss_popts[ext], pedestal, gain,
                                             centers_e, counts_e, electron_fit_mode)
        sigma_e, N1 = popt_e[0], popt_e[4]

        if yscale != 'linear':
            ax.set_yscale(yscale)
        ax.stairs(_bar_heights(counts_e, yscale), edges_e, fill=True, color='#4e117d')

        curve_x_e = _fit_curve_x(zero_one_range_e[0], zero_one_range_e[1], len(counts_e))
        ax.plot(curve_x_e, double_gauss(curve_x_e, *popt_e), '#8dde1b',
                label=_dark_current_fit_label(sigma_e, gain, N1))

        # Each dark-current method actually present in dark_current_rows gets its own,
        # independent legend entry -- these are not mutually exclusive.
        dc_count = dark_current_rows[ext].get('dark_current_count_e_per_pix_day')
        if dc_count is not None:
            # 'count' method was used: draw its +/- count_nsigma * sigma window about the
            # 1 e- charge, computed in ADU exactly as the method does, then converted to
            # electrons so the lines match the pixels actually counted (independent of
            # any refit). Label it with the method's dark current.
            popt = double_gauss_popts[ext]
            sigma_adu = popt[0]
            center_adu = popt[2] if count_center == 'mu1' else pedestal + gain
            lo_e = (center_adu - count_nsigma * sigma_adu - pedestal) / gain
            hi_e = (center_adu + count_nsigma * sigma_adu - pedestal) / gain
            window_label = (rf'count (window $\pm${count_nsigma:g}$\,\sigma$)'
                            + '\n' + r'dark current = %.3g $e^-$/pix/day' % dc_count)
            ax.axvline(lo_e, color='k', linestyle=':', linewidth=1.2, label=window_label)
            ax.axvline(hi_e, color='k', linestyle=':', linewidth=1.2)

        dc_integrate = dark_current_rows[ext].get('dark_current_integrate_e_per_pix_day')
        if dc_integrate is not None:
            # 'integrate' method was used: shade the area under the fitted one-electron
            # Gaussian component (reusing popt_e/curve_x_e from the fit curve above, so
            # the shading matches it exactly) and label it with the method's dark current.
            one_e_curve = popt_e[4] * np.exp(-(curve_x_e - popt_e[2])**2 / (2 * popt_e[0]**2))
            integrate_label = (r'integrate (1$e^-$ Gaussian)'
                               + '\n' + r'dark current = %.3g $e^-$/pix/day' % dc_integrate)
            ax.fill_between(curve_x_e, one_e_curve, color='#ff7f0e', alpha=0.4,
                            label=integrate_label)

        dc_weighted = dark_current_rows[ext].get('dark_current_weighted_e_per_pix_day')
        if dc_weighted is not None:
            # 'weighted' method was used: report its dark current and formula as a
            # legend-only entry (no corresponding visual element).
            ax.plot([], [], ' ',
                    label=(r'weighted (SER = $N_1$ / $(N_1 + N_0)$)' \
                           + '\n' + 'dark current = %.3g $e^-$/pix/day' % dc_weighted))

        ax.set_xlim(zero_one_range_e[0], zero_one_range_e[1])
        ax.legend(loc='upper right', fontsize=fontsize - 2)

    if save_plots:
        output_path = fig_path / f'{base_name}.jpeg'
        plt.savefig(str(output_path), dpi=dpi)
        print(f'Saved plot to {output_path}')
    _finish_fig(show_plots)


def plot_charge_per_column(data_ext, n_std_to_mask=1.5, fit_cols_ext=None,
                           figsize=_SUBPLOTS_FIGSIZE, additional_title='', show_titles=True,
                           nimages=1, verbose=False, save_plots=False, show_plots=True,
                           fig_path='./', file='charge_per_column', dpi=350):
    """Median charge per column for each extension, as a 2x2 grid of scatter plots.

    Intended for the *raw* (pre-pedestal-subtraction) data, so the per-column medians
    reveal which columns carry anomalous charge. A column is flagged "hot" (drawn red)
    when its median sits at least ``n_std_to_mask`` biweight standard deviations from
    the biweight location of the extension -- the same robust threshold the pedestal
    subtraction uses to mask outliers. Every column is plotted; ``fit_cols_ext``, if
    given, is a per-extension ``(c0, c1)`` slice (or None) marking which columns the
    zero/one fit keeps -- the columns *outside* it are shaded light grey, so the plot
    shows the full frame while making the masked-out region visible.
    """
    fig_path = Path(fig_path)
    base_name = Path(file).stem + '_charge_per_column' if file != 'charge_per_column' else file

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    if show_titles:
        fig.suptitle(f'{additional_title}Median Charge per Column (Nimages = {nimages})')

    for ext, data in enumerate(data_ext):
        ax = axes.flat[ext]
        data = np.asarray(data)
        col_idx = np.arange(data.shape[1])
        median_charge = np.median(data, axis=0)

        flat = data.flatten()
        loc = biweight_location(flat, ignore_nan=True)
        scale = np.sqrt(biweight_midvariance(flat, ignore_nan=True))
        hot = np.abs(median_charge - loc) >= n_std_to_mask * scale

        ax.scatter(col_idx[~hot], median_charge[~hot], s=1)
        ax.scatter(col_idx[hot], median_charge[hot], s=1, color='red')

        # Shade the columns excluded by fit_cols (Python-slice semantics handle negative
        # or None endpoints). Each maximal run of excluded columns is one grey span, so a
        # left and/or right cut both show; nothing is drawn when all columns are kept.
        fc = fit_cols_ext[ext] if fit_cols_ext is not None else None
        if fc is not None:
            kept = np.zeros(data.shape[1], dtype=bool)
            kept[fc[0]:fc[1]] = True
            edges = np.diff(np.concatenate(([0], (~kept).astype(np.int8), [0])))
            for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                ax.axvspan(start - 0.5, end - 0.5, color='gray', alpha=0.15, linewidth=0)

        if show_titles:
            ax.set_title(f'EXT {ext + 1}')
        if ext % 2 == 0:        # left column of the grid
            ax.set_ylabel('Median Charge (ADU)')
        if ext // 2 == 1:       # bottom row of the grid
            ax.set_xlabel('Column')

        if verbose:
            pct = 100 * hot.sum() / len(hot) if len(hot) else 0.0
            print(f'EXT {ext + 1}: {int(hot.sum())} hot columns ({pct:.1f}%) '
                  f'(median charge >= {n_std_to_mask} SDs from pedestal location): '
                  f'{col_idx[hot].tolist()}')

    if save_plots:
        output_path = fig_path / f'{base_name}.jpeg'
        plt.savefig(str(output_path), dpi=dpi)
        print(f'Saved plot to {output_path}')
    _finish_fig(show_plots)
