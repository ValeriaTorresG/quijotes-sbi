'''Figures in the project's notebook style: LaTeX text, royalblue/darkorange,
dashed black references, light grids.

``SBIPlotter()`` applies the style to the global rcParams (like the notebooks
do); pass ``style=False`` to keep your own. Every method returns a Figure.
Most methods take plain arrays or an ``EvaluationResult``, so figures can be
redrawn from saved results (see ``replot.py``) without retraining anything.
'''
from __future__ import annotations

import shutil
from typing import Mapping

import numpy as np

from .config import ALL_PARAMETER_LABELS, PARAMETER_LABELS, PARAMETER_NAMES

PRIMARY, SECONDARY, REFERENCE = 'royalblue', 'darkorange', 'black'
COLORS = ('royalblue', 'darkorange', 'forestgreen', 'crimson', 'mediumpurple', 'saddlebrown', 'hotpink',
          'gray', 'olive', 'darkcyan')
SERIES = COLORS  # backwards-compatible name
GRID = dict(lw=0.3)


def latex_available() -> bool:
    return shutil.which('latex') is not None and shutil.which('dvipng') is not None


def apply_style(usetex: bool | None = None, dpi: int = 360):
    '''Notebook style: LaTeX (if installed) with amsmath, high dpi.'''
    import matplotlib
    usetex = latex_available() if usetex is None else usetex
    matplotlib.rcParams['figure.dpi'] = dpi
    matplotlib.rcParams['savefig.dpi'] = dpi
    matplotlib.rcParams['text.usetex'] = usetex
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def tex(text) -> str:
    '''Escape free text (experiment names, file labels) for LaTeX when it is on.'''
    import matplotlib
    text = str(text)
    if not matplotlib.rcParams['text.usetex']:
        return text
    for char in ('\\', '_', '%', '&', '#'):
        text = text.replace(char, '\\' + char if char != '\\' else r'\textbackslash{}')
    return text


def pct(value) -> str:
    import matplotlib
    return f'{value}\\%' if matplotlib.rcParams['text.usetex'] else f'{value}%'


def _labels(names, override=None):
    if override is not None:
        if len(override) != len(names):
            raise ValueError('labels must match the number of parameters.')
        return tuple(override)
    return tuple(ALL_PARAMETER_LABELS.get(n, tex(n)) for n in names)


def _grid(n_panels, ncols=None, panel=(4.3, 3.5)):
    import matplotlib.pyplot as plt
    ncols = ncols or min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_panels:]:
        fig.delaxes(ax)
    for ax in axes[:n_panels]:
        ax.grid(**GRID)
    return fig, axes[:n_panels]


def _ellipse(ax, mean, cov, delta_chi2, **kwargs):
    from matplotlib.patches import Ellipse
    values, vectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    width, height = 2 * np.sqrt(delta_chi2 * values[::-1])
    ax.add_patch(Ellipse(mean, width, height, angle=angle, fill=False, **kwargs))


def _density_levels(hist, fractions=(0.95, 0.68)):
    flat = np.sort(hist.ravel())[::-1]
    cumulative = np.cumsum(flat) / flat.sum()
    return sorted({flat[min(np.searchsorted(cumulative, f), len(flat) - 1)] for f in fractions})


def marginal_covariance(fisher, names):
    '''Fisher-marginalized covariance of ``names`` (object with .parameters and .fisher).'''
    index = [list(fisher.parameters).index(n) for n in names]
    return np.linalg.inv(fisher.fisher)[np.ix_(index, index)]


class SBIPlotter:
    def __init__(self, labels=None, style: bool = True, usetex: bool | None = None):
        self.labels = None if labels is None else tuple(labels)
        if style:
            apply_style(usetex)

    # ------------------------------------------------------------------ test set
    def prediction(self, evaluation, title=''):
        '''Posterior mean and 68 % interval against the truth, one panel per parameter.'''
        ev = evaluation
        labels = _labels(ev.parameter_names, self.labels)
        summary = ev.summary
        fig, axes = _grid(len(labels))
        for j, (ax, label) in enumerate(zip(axes, labels)):
            ax.vlines(ev.truth[:, j], ev.low68[:, j], ev.high68[:, j], color=PRIMARY, alpha=0.25)
            ax.scatter(ev.truth[:, j], ev.means[:, j], s=12, color=PRIMARY, alpha=0.7)
            if ev.prior_low is not None:
                limits = [ev.prior_low[j], ev.prior_high[j]]
            else:
                limits = [min(ev.truth[:, j].min(), ev.low68[:, j].min()),
                          max(ev.truth[:, j].max(), ev.high68[:, j].max())]
            ax.plot(limits, limits, 'k--')
            row = summary.iloc[j]
            ax.text(0.04, 0.96, f'$R^2 = {row.R2:.2f}$\nRMSE $= {row.RMSE:.3g}$\n'
                                f'cov$_{{68}} = {row.coverage_68:.2f}$',
                    transform=ax.transAxes, va='top', fontsize=9)
            ax.set(xlabel='True value', ylabel='Mean posterior', title=label)
        suffix = ' (MN, Gaussian intervals)' if ev.interval_method == 'gaussian_from_moments' else ''
        if title or suffix:
            fig.suptitle(tex(title) + suffix)
        fig.tight_layout()
        return fig

    def coverage(self, evaluation, title=''):
        import matplotlib.pyplot as plt
        labels = _labels(evaluation.parameter_names, self.labels)
        summary = evaluation.summary
        positions = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.grid(**GRID)
        ax.bar(positions - 0.18, summary['coverage_68'], width=0.36, label=f'{pct(68)} interval', color=PRIMARY)
        ax.bar(positions + 0.18, summary['coverage_95'], width=0.36, label=f'{pct(95)} interval', color=SECONDARY)
        ax.axhline(0.68, color=PRIMARY, linestyle='--')
        ax.axhline(0.95, color=SECONDARY, linestyle='--')
        ax.set_xticks(positions, labels)
        ax.set(ylim=(0, 1), ylabel='Fraction containing the true value')
        if title:
            ax.set_title(tex(title))
        ax.legend(loc='lower right')
        fig.tight_layout()
        return fig

    def diagnostics(self, result):
        '''(prediction_figure, coverage_figure) of an ExperimentResult.'''
        title = result.name or ' + '.join(result.fit.split.data.envs)
        return self.prediction(result.evaluation, title), self.coverage(result.evaluation, title)

    def calibration(self, evaluation, title='Calibration'):
        '''Empirical vs nominal coverage of central intervals (band: 95 % binomial).'''
        curve = evaluation.coverage_curve()
        labels = _labels(evaluation.parameter_names, self.labels)
        fig, axes = _grid(len(labels))
        for ax, name, label in zip(axes, evaluation.parameter_names, labels):
            c = curve[curve.param == name]
            ax.fill_between(c.level, c.ci_low, c.ci_high, color=PRIMARY, alpha=0.2, lw=0)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.plot(c.level, c.empirical, color=PRIMARY, marker='o', ms=3)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Nominal coverage', ylabel='Empirical coverage', title=label)
        fig.suptitle(tex(title))
        fig.tight_layout()
        return fig

    def sbc(self, evaluation, bins=20, title='SBC ranks'):
        '''PIT histograms; flat within the grey band (99 %) means calibrated marginals.'''
        from scipy import stats
        labels = _labels(evaluation.parameter_names, self.labels)
        n = len(evaluation.pit)
        low, high = stats.binom.ppf([0.005, 0.995], n, 1 / bins)
        fig, axes = _grid(len(labels))
        for j, (ax, label) in enumerate(zip(axes, labels)):
            ax.axhspan(low, high, color='gray', alpha=0.2, lw=0)
            ax.axhline(n / bins, color=REFERENCE, linestyle='--')
            ax.hist(evaluation.pit[:, j], bins=bins, range=(0, 1), color=PRIMARY, alpha=0.65)
            ax.set(xlim=(0, 1), xlabel=r'$P(\theta < \theta_{\rm true} \mid x)$', ylabel='Count', title=label)
        fig.suptitle(tex(title))
        fig.tight_layout()
        return fig

    def tarp(self, evaluation, title='TARP'):
        import matplotlib.pyplot as plt
        alpha, ecp = evaluation.tarp_curve()
        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        ax.grid(**GRID)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(alpha, ecp, color=PRIMARY, lw=2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r'Credibility level $\alpha$', ylabel='Expected coverage')
        ax.set_title(f'{tex(title)}: max $|{{\\rm ECP}} - \\alpha| = {np.abs(ecp - alpha).max():.3f}$')
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ posteriors
    def corner(self, samples, names, truth=None, fisher=None, title=''):
        '''Corner plot of posterior samples; ``fisher=(mean, cov)`` overlays 68/95 % ellipses.'''
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from scipy.ndimage import gaussian_filter
        labels = _labels(names, self.labels)
        p = len(names)
        fig, axes = plt.subplots(p, p, figsize=(2.3 * p, 2.3 * p), squeeze=False)
        ranges = [np.quantile(samples[:, j], [0.001, 0.999]) for j in range(p)]
        if fisher is not None:
            sigma = np.sqrt(np.diag(fisher[1]))
            ranges = [(min(r[0], fisher[0][j] - 3.5 * sigma[j]), max(r[1], fisher[0][j] + 3.5 * sigma[j]))
                      for j, r in enumerate(ranges)]
        for i in range(p):
            for j in range(p):
                ax = axes[i, j]
                if j > i:
                    ax.set_visible(False)
                    continue
                if i == j:
                    ax.hist(samples[:, j], bins=50, range=ranges[j], density=True, color=PRIMARY, alpha=0.65)
                    if fisher is not None:
                        grid = np.linspace(*ranges[j], 200)
                        s = np.sqrt(fisher[1][j, j])
                        ax.plot(grid, np.exp(-0.5 * ((grid - fisher[0][j]) / s) ** 2) / (s * np.sqrt(2 * np.pi)),
                                color=SECONDARY, lw=1.5)
                    ax.set_yticks([])
                else:
                    hist, xe, ye = np.histogram2d(samples[:, j], samples[:, i], bins=40,
                                                  range=[ranges[j], ranges[i]])
                    hist = gaussian_filter(hist, 1.0)
                    levels = _density_levels(hist)
                    if len(levels) == 2 and levels[0] < levels[1]:
                        from matplotlib.colors import to_rgba
                        ax.contourf(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), hist.T,
                                    levels=[levels[0], levels[1], hist.max()],
                                    colors=[to_rgba(PRIMARY, 0.3), to_rgba(PRIMARY, 0.6)])
                    if fisher is not None:
                        cov = fisher[1][np.ix_([j, i], [j, i])]
                        for d2, lw in ((2.30, 1.5), (6.18, 1.0)):
                            _ellipse(ax, (fisher[0][j], fisher[0][i]), cov, d2, color=SECONDARY, lw=lw)
                    ax.set_xlim(ranges[j])
                    ax.set_ylim(ranges[i])
                    ax.grid(**GRID)
                if truth is not None:
                    ax.axvline(truth[j], color=REFERENCE, ls='--', lw=1)
                    if i != j:
                        ax.axhline(truth[i], color=REFERENCE, ls='--', lw=1)
                if i == p - 1:
                    ax.set_xlabel(labels[j])
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i])
                elif j > 0:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=8)
        handles = [Line2D([], [], color=PRIMARY, lw=6, alpha=0.6)]
        names_legend = ['Posterior']
        if fisher is not None:
            handles.append(Line2D([], [], color=SECONDARY))
            names_legend.append(f'Fisher ({pct(68)}, {pct(95)})')
        if truth is not None:
            handles.append(Line2D([], [], color=REFERENCE, ls='--'))
            names_legend.append('True value')
        fig.legend(handles, names_legend, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        if title:
            fig.suptitle(tex(title))
        fig.tight_layout()
        return fig

    def posterior(self, result, idx: int = 0, n_samples: int = 50_000, seed: int = 123):
        '''Corner plot of the posterior of one held-out cube.'''
        if not hasattr(result.fit, 'sample'):
            raise ValueError('Moment Networks provide moments, not posterior samples or densities.')
        if not 0 <= idx < len(result.fit.split.test_rows):
            raise IndexError('idx must index a held-out cube.')
        samples = result.fit.sample(result.fit.split.x_test[idx], n_samples, seed).numpy()
        cube = result.fit.split.test_cube_ids[idx]
        return self.corner(samples, result.evaluation.parameter_names, truth=result.evaluation.truth[idx],
                           title=f'{result.name}: test cube {cube}')

    def fiducial_posterior(self, result, x, n_samples: int = 50_000, seed: int = 123):
        '''NPE posterior of one fiducial realization against the Fisher ellipses.'''
        names = result.evaluation.parameter_names
        samples = result.fit.sample(x, n_samples, seed).numpy()
        truth = result.fiducial.truth[0] if result.fiducial is not None else None
        fisher = None
        if result.fisher is not None and set(names) <= set(result.fisher.parameters):
            fisher = (np.array([result.fisher.fiducial[n] for n in names]), marginal_covariance(result.fisher, names))
        return self.corner(samples, names, truth=truth, fisher=fisher, title=f'{result.name}: fiducial realization')

    def fiducial_scatter(self, evaluation, title='Fiducial realizations'):
        '''z = (posterior mean - fiducial) / posterior std over fiducial realizations.'''
        if hasattr(evaluation, 'fiducial'):  # ExperimentResult (old call signature)
            title, evaluation = f'{evaluation.name}: fiducial realizations', evaluation.fiducial
        labels = _labels(evaluation.parameter_names, self.labels)
        fig, axes = _grid(len(labels))
        grid = np.linspace(-4, 4, 200)
        for j, (ax, label) in enumerate(zip(axes, labels)):
            z = (evaluation.means[:, j] - evaluation.truth[:, j]) / evaluation.stds[:, j]
            ax.hist(z, bins=30, range=(-4, 4), density=True, color=PRIMARY, alpha=0.65)
            ax.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), 'k--')
            ax.set(xlabel=r'$(\bar\theta - \theta_{\rm fid}) / \sigma_\theta$', ylabel='PDF',
                   title=f'{label}: $\\langle z \\rangle = {z.mean():.2f}$')
        fig.suptitle(tex(title))
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ comparisons
    @staticmethod
    def error_table(results: Mapping):
        '''DataFrame (experiment, param, npe_std, fisher_sigma) from ExperimentResults.'''
        import pandas as pd
        rows = []
        for name, r in results.items():
            names = r.evaluation.parameter_names
            fisher = dict(zip(r.fisher.parameters, r.fisher.sigma)) if r.fisher is not None else {}
            for j, p in enumerate(names):
                rows.append(dict(experiment=name, param=p,
                                 npe_std=r.fiducial.stds[:, j].mean() if r.fiducial is not None else np.nan,
                                 fisher_sigma=fisher.get(p, np.nan)))
        return pd.DataFrame(rows)

    def compare_errors(self, table, parameters=PARAMETER_NAMES, method_label='NPE'):
        '''Per-parameter 1-sigma errors across experiments: posterior at the fiducial vs Fisher.'''
        if not hasattr(table, 'columns'):
            table = self.error_table(table)
        names = list(dict.fromkeys(table.experiment))
        labels = _labels(parameters, self.labels)
        fig, axes = _grid(len(parameters), panel=(4.3, 0.35 * len(names) + 1.8))
        y = np.arange(len(names))
        for j, (ax, parameter, label) in enumerate(zip(axes, parameters, labels)):
            rows = table[table.param == parameter].set_index('experiment')
            ax.barh(y - 0.18, [rows.npe_std.get(n, np.nan) for n in names], height=0.36, color=PRIMARY,
                    label=rf'{method_label} $\langle\sigma\rangle$ (fiducial)')
            ax.barh(y + 0.18, [rows.fisher_sigma.get(n, np.nan) for n in names], height=0.36, color=SECONDARY,
                    label=r'Fisher $\sigma$')
            ax.set_xscale('log')
            ax.set_yticks(y, [tex(n) for n in names] if j % 3 == 0 else [''] * len(names))
            ax.invert_yaxis()
            ax.set_title(label)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper center', ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return fig

    def compare_metrics(self, comparison, metric='RMSE'):
        '''One panel per parameter: a test-set metric for every experiment (and method).'''
        params = list(dict.fromkeys(comparison.param))
        labels = _labels(params, self.labels)
        experiments = list(dict.fromkeys(comparison.experiment))
        methods = list(dict.fromkeys(comparison.method)) if 'method' in comparison else ['npe']
        if 'method' not in comparison:
            comparison = comparison.assign(method='npe')
        fig, axes = _grid(len(params), panel=(4.3, 0.35 * len(experiments) + 1.8))
        y = np.arange(len(experiments))
        height = 0.8 / len(methods)
        for j, (ax, param, label) in enumerate(zip(axes, params, labels)):
            for m, method in enumerate(methods):
                rows = comparison[(comparison.param == param) & (comparison.method == method)].set_index('experiment')
                ax.barh(y + (m - (len(methods) - 1) / 2) * height, [rows[metric].get(e, np.nan) for e in experiments],
                        height=height * 0.9, color=COLORS[m % len(COLORS)], label=tex(method.upper()))
            if metric.startswith('coverage_'):
                ax.axvline(float(metric.split('_')[1]) / 100, color=REFERENCE, ls='--')
            ax.set_yticks(y, [tex(e) for e in experiments] if j % 3 == 0 else [''] * len(experiments))
            ax.invert_yaxis()
            ax.set(title=label, xlabel=tex(metric))
        if len(methods) > 1:
            handles, legend_labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='upper center', ncol=len(methods))
        fig.tight_layout(rect=(0, 0, 1, 0.93 if len(methods) > 1 else 1))
        return fig

    # ------------------------------------------------------------------ Fisher
    def fisher_ellipses(self, fishers: Mapping, parameters=None):
        '''Triangle of 68 % Fisher ellipses (objects with .parameters, .fisher, .fiducial).'''
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        names = list(fishers)
        if len(names) > len(COLORS):
            raise ValueError(f'At most {len(COLORS)} experiments per figure (got {len(names)}); split them.')
        parameters = tuple(parameters or next(iter(fishers.values())).parameters)
        labels = _labels(parameters, self.labels)
        p = len(parameters)
        covs = {name: (np.array([f.fiducial[q] for q in parameters]), marginal_covariance(f, parameters))
                for name, f in fishers.items()}
        span = np.max([np.sqrt(np.diag(c)) for _, c in covs.values()], axis=0) * 3.2
        center = next(iter(covs.values()))[0]
        fig, axes = plt.subplots(p, p, figsize=(2.3 * p, 2.3 * p), squeeze=False)
        for i in range(p):
            for j in range(p):
                ax = axes[i, j]
                if j > i:
                    ax.set_visible(False)
                    continue
                for k, (mean, cov) in enumerate(covs.values()):
                    if i == j:
                        grid = np.linspace(center[j] - span[j], center[j] + span[j], 300)
                        ax.plot(grid, np.exp(-0.5 * ((grid - mean[j]) / np.sqrt(cov[j, j])) ** 2), color=COLORS[k])
                    else:
                        _ellipse(ax, (mean[j], mean[i]), cov[np.ix_([j, i], [j, i])], 2.30, color=COLORS[k], lw=1.5)
                ax.set_xlim(center[j] - span[j], center[j] + span[j])
                if i != j:
                    ax.set_ylim(center[i] - span[i], center[i] + span[i])
                    ax.grid(**GRID)
                else:
                    ax.set_yticks([])
                if i == p - 1:
                    ax.set_xlabel(labels[j])
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i])
                elif j > 0:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=8)
        fig.legend([Line2D([], [], color=COLORS[k]) for k in range(len(names))], [tex(n) for n in names],
                   loc='upper right', bbox_to_anchor=(0.98, 0.98))
        fig.suptitle(f'Fisher forecast, {pct(68)} contours')
        fig.tight_layout()
        return fig

    def posterior_comparison(self, entries: Mapping, names, truth=None, title='', levels=(0.68, 0.95),
                             smooth=1.6):
        '''Several posteriors (filled 68 %, outlined 95 %) with their Fisher ellipses, one colour each.

        ``entries``: label -> (samples [n, p], fisher) where fisher is None or an
        object with .parameters, .fisher and .fiducial. Draw the widest first.
        '''
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from scipy.ndimage import gaussian_filter
        if len(entries) > len(COLORS):
            raise ValueError(f'At most {len(COLORS)} posteriors per figure.')
        labels = _labels(names, self.labels)
        p = len(names)
        colors = dict(zip(entries, COLORS))
        overlays = {k: (np.array([f.fiducial[n] for n in names]), marginal_covariance(f, names)) if f is not None
                    else None for k, (_, f) in entries.items()}
        low = np.min([np.quantile(s, 0.002, axis=0) for s, _ in entries.values()], axis=0)
        high = np.max([np.quantile(s, 0.998, axis=0) for s, _ in entries.values()], axis=0)
        ranges = list(zip(low, high))
        fig, axes = plt.subplots(p, p, figsize=(2.3 * p, 2.3 * p), squeeze=False)
        for i in range(p):
            for j in range(p):
                ax = axes[i, j]
                if j > i:
                    ax.set_visible(False)
                    continue
                for key, (samples, _) in entries.items():
                    color, overlay = colors[key], overlays[key]
                    if i == j:
                        ax.hist(samples[:, j], bins=40, range=ranges[j], density=True, histtype='step',
                                color=color, lw=1.5)
                        if overlay is not None:
                            grid = np.linspace(*ranges[j], 300)
                            s = np.sqrt(overlay[1][j, j])
                            ax.plot(grid, np.exp(-0.5 * ((grid - overlay[0][j]) / s) ** 2) / (s * np.sqrt(2 * np.pi)),
                                    color=color, ls='--', lw=1.2)
                        continue
                    hist, xe, ye = np.histogram2d(samples[:, j], samples[:, i], bins=40,
                                                  range=[ranges[j], ranges[i]])
                    hist = gaussian_filter(hist, smooth)
                    contour_levels = _density_levels(hist, fractions=tuple(sorted(levels, reverse=True)))
                    if len(contour_levels) == 2 and contour_levels[0] < contour_levels[1]:
                        xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                        ax.contourf(xc, yc, hist.T, levels=[contour_levels[1], hist.max()],
                                    colors=[to_rgba(color, 0.35)])
                        ax.contour(xc, yc, hist.T, levels=contour_levels, colors=[color],
                                   linewidths=[0.9, 1.4])
                    if overlay is not None:
                        _ellipse(ax, (overlay[0][j], overlay[0][i]), overlay[1][np.ix_([j, i], [j, i])], 2.30,
                                 color=color, ls='--', lw=1.2)
                ax.set_xlim(ranges[j])
                if i == j:
                    ax.set_yticks([])
                else:
                    ax.set_ylim(ranges[i])
                    ax.grid(**GRID)
                if truth is not None:
                    ax.axvline(truth[j], color=REFERENCE, ls=':', lw=1)
                    if i != j:
                        ax.axhline(truth[i], color=REFERENCE, ls=':', lw=1)
                if i == p - 1:
                    ax.set_xlabel(labels[j])
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i])
                elif j > 0:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=8)
        handles = [Patch(facecolor=to_rgba(colors[k], 0.4), edgecolor=colors[k]) for k in entries]
        names_legend = [tex(k) for k in entries]
        handles += [Patch(facecolor=to_rgba('gray', 0.35), edgecolor='gray'), Line2D([], [], color='gray', lw=0.9),
                    Line2D([], [], color='gray', ls='--')]
        names_legend += [f'Posterior {pct(68)}', f'Posterior {pct(95)}', f'Fisher {pct(68)}']
        if truth is not None:
            handles.append(Line2D([], [], color=REFERENCE, ls=':'))
            names_legend.append('True value')
        fig.legend(handles, names_legend, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        if title:
            fig.suptitle(tex(title))
        fig.tight_layout()
        return fig

    def posterior_panels(self, entries: Mapping, names, pairs=(('Omega_m', 'sigma8'), ('n_s', 'sigma8')),
                         truth=None, levels=(0.68,), title='', smooth=1.6, bounds=None):
        '''One column per observable, one row per parameter pair; shared limits within each row.

        ``entries``: label -> (samples [n, len(names)], fisher or None). SBI is a
        solid blue contour with a faint fill; Fisher a dashed red ellipse.
        ``bounds``: {param: (low, high)} to clip the axes (e.g. the prior box).
        '''
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from scipy.ndimage import gaussian_filter
        names = list(names)
        missing = {p for pair in pairs for p in pair} - set(names)
        if missing:
            raise ValueError(f'Unknown parameters {sorted(missing)}; choose from {names}.')
        sbi_color, fisher_color = PRIMARY, 'crimson'
        chi2 = {0.68: 2.30, 0.95: 6.18}
        levels = tuple(sorted(levels, reverse=True))
        widths = {lvl: (1.6 if lvl == min(levels) else 0.9) for lvl in levels}
        nrows, ncols = len(pairs), len(entries)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.1 * nrows), squeeze=False,
                                 sharex='row', sharey='row')
        for r, (px, py) in enumerate(pairs):
            jx, jy = names.index(px), names.index(py)
            # Shared limits per row: union of every SBI posterior and Fisher ellipse.
            low = np.full(2, np.inf)
            high = np.full(2, -np.inf)
            for samples, fisher in entries.values():
                q = np.quantile(samples[:, [jx, jy]], [0.003, 0.997], axis=0)
                low, high = np.minimum(low, q[0]), np.maximum(high, q[1])
                if fisher is not None:
                    center = np.array([fisher.fiducial[px], fisher.fiducial[py]])
                    sigma = np.sqrt(np.diag(marginal_covariance(fisher, [px, py])))
                    low, high = np.minimum(low, center - 2.6 * sigma), np.maximum(high, center + 2.6 * sigma)
            pad = 0.04 * (high - low)
            low, high = low - pad, high + pad
            if bounds is not None:
                for k, p in enumerate((px, py)):
                    if p in bounds:
                        b_low, b_high = bounds[p]
                        margin = 0.03 * (b_high - b_low)
                        low[k], high[k] = max(low[k], b_low - margin), min(high[k], b_high + margin)
            for c, (label, (samples, fisher)) in enumerate(entries.items()):
                ax = axes[r, c]
                hist, xe, ye = np.histogram2d(samples[:, jx], samples[:, jy], bins=45,
                                              range=[[low[0], high[0]], [low[1], high[1]]])
                hist = gaussian_filter(hist, smooth)
                xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                values = _density_levels(hist, fractions=levels)
                if len(values) == len(levels):
                    ax.contourf(xc, yc, hist.T, levels=[min(values), hist.max()], colors=[to_rgba(sbi_color, 0.15)])
                    ax.contour(xc, yc, hist.T, levels=values, colors=[sbi_color],
                               linewidths=[widths[lvl] for lvl in sorted(levels)][::-1])
                if fisher is not None:
                    center = (fisher.fiducial[px], fisher.fiducial[py])
                    cov = marginal_covariance(fisher, [px, py])
                    for lvl in levels:
                        _ellipse(ax, center, cov, chi2[lvl], color=fisher_color, ls='--', lw=widths[lvl])
                if truth is not None:
                    ax.plot(truth[jx], truth[jy], '+', color=REFERENCE, ms=11, mew=1.5)
                ax.set_xlim(low[0], high[0])
                ax.set_ylim(low[1], high[1])
                ax.grid(**GRID)
                if r == 0:
                    ax.set_title(tex(label))
                ax.set_xlabel(ALL_PARAMETER_LABELS.get(px, tex(px)))
                if c == 0:
                    ax.set_ylabel(ALL_PARAMETER_LABELS.get(py, tex(py)))
        region = ', '.join(pct(int(round(100 * lvl))) for lvl in sorted(levels))
        handles = [Patch(facecolor=to_rgba(sbi_color, 0.15), edgecolor=sbi_color, lw=1.6),
                   Line2D([], [], color=fisher_color, ls='--', lw=1.6),
                   Line2D([], [], color=REFERENCE, marker='+', ls='none', ms=11, mew=1.5)]
        fig.legend(handles, [f'SBI, {region} region', f'Fisher, {region} region', 'True value'],
                   loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0))
        if title:
            fig.suptitle(tex(title))
        fig.tight_layout(rect=(0, min(0.12, 0.14 / nrows), 1, 1))
        return fig

    def correlation(self, fisher, title=None):
        '''Correlation matrix of the features at the fiducial (.covariance, .blocks, .block_sizes).'''
        import matplotlib.pyplot as plt
        covariance = np.asarray(fisher.covariance)
        d = np.sqrt(np.diag(covariance))
        corr = covariance / np.outer(d, d)
        fig, ax = plt.subplots(figsize=(6.5, 5.6))
        image = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        sizes = list(getattr(fisher, 'block_sizes', None) or [len(d)])
        edges = np.cumsum([0] + sizes)
        for e in edges[1:-1]:
            ax.axhline(e - 0.5, color=REFERENCE, lw=0.6)
            ax.axvline(e - 0.5, color=REFERENCE, lw=0.6)
        if len(sizes) == len(fisher.blocks):
            centers = 0.5 * (edges[1:] + edges[:-1]) - 0.5
            ax.set_xticks(centers, [tex(b) for b in fisher.blocks], rotation=45, ha='right', fontsize=8)
            ax.set_yticks(centers, [tex(b) for b in fisher.blocks], fontsize=8)
        fig.colorbar(image, ax=ax, label='Correlation coefficient', shrink=0.85)
        ax.set_title(tex(title or 'Feature correlations (fiducial)'))
        fig.tight_layout()
        return fig

    def convergence(self, table, title='Fisher convergence'):
        names = list(dict.fromkeys(table.param))
        fig, axes = _grid(2, ncols=2, panel=(4.8, 3.8))
        for ax, kind in zip(axes, ('covariance', 'derivative')):
            t = table[table.kind == kind]
            for k, name in enumerate(names):
                rows = t[t.param == name]
                ax.plot(rows.n, rows.ratio_to_full, marker='o', ms=4, color=COLORS[k],
                        label=ALL_PARAMETER_LABELS.get(name, tex(name)))
            ax.axhline(1, color=REFERENCE, ls='--')
            ax.set(xlabel=f'Realizations for the {kind}', ylabel=r'$\sigma / \sigma_{\rm all}$', title=kind.capitalize())
        axes[0].legend(fontsize=9)
        fig.suptitle(tex(title))
        fig.tight_layout()
        return fig


__all__ = ['SBIPlotter', 'COLORS', 'SERIES', 'PARAMETER_LABELS', 'apply_style', 'latex_available', 'tex']
