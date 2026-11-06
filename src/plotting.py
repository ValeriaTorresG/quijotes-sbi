'''Figures for validation and comparison. Every method returns a Figure and
never changes global rcParams (call ``fig.savefig`` or display it yourself).'''
from __future__ import annotations

from typing import Mapping

import numpy as np

from .config import ALL_PARAMETER_LABELS, PARAMETER_LABELS, PARAMETER_NAMES

# Categorical slots in fixed order (validated for colour-vision deficiency on
# adjacent pairs); ink and grid tokens for a light surface.
SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, INK_2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BAND = '#cde2fb'


def _style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)


def _labels(names, override=None):
    if override is not None:
        if len(override) != len(names):
            raise ValueError('labels must match the number of parameters.')
        return tuple(override)
    return tuple(ALL_PARAMETER_LABELS.get(n, n) for n in names)


def _grid(n_panels, width=4.0, height=3.4, ncols=None):
    import matplotlib.pyplot as plt
    ncols = ncols or min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    for ax in axes.ravel()[n_panels:]:
        ax.set_visible(False)
    for ax in axes.ravel()[:n_panels]:
        _style(ax)
    return fig, axes.ravel()


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


class SBIPlotter:
    def __init__(self, labels=None):
        self.labels = None if labels is None else tuple(labels)

    def _names(self, result):
        return result.evaluation.parameter_names

    # ------------------------------------------------------------------ test set
    def diagnostics(self, result):
        '''(prediction_figure, coverage_figure) for the held-out cubes.'''
        import matplotlib.pyplot as plt
        ev = result.evaluation
        labels = _labels(ev.parameter_names, self.labels)
        summary = ev.summary
        fig, axes = _grid(len(labels))
        for j, (ax, label) in enumerate(zip(axes, labels)):
            lo, hi = ev.means[:, j] - ev.low68[:, j], ev.high68[:, j] - ev.means[:, j]
            ax.errorbar(ev.truth[:, j], ev.means[:, j], yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt='o',
                        ms=3, color=SERIES[0], ecolor=SERIES[0], elinewidth=0.8, alpha=0.6)
            limits = [ev.truth[:, j].min(), ev.truth[:, j].max()]
            if ev.prior_low is not None:
                limits = [ev.prior_low[j], ev.prior_high[j]]
            ax.plot(limits, limits, color=MUTED, linestyle='--', linewidth=1)
            row = summary.iloc[j]
            ax.set_title(label, color=INK)
            ax.text(0.03, 0.97, f'R$^2$ = {row.R2:.2f}\nRMSE = {row.RMSE:.3g}\ncov68 = {row.coverage_68:.2f}',
                    transform=ax.transAxes, va='top', fontsize=8.5, color=INK_2)
            ax.set(xlabel='True value', ylabel='Posterior mean (68 % interval)')
        suffix = ' (MN; Gaussian moment intervals)' if ev.interval_method == 'gaussian_from_moments' else ''
        fig.suptitle(f'{result.name or " + ".join(result.fit.split.data.envs)}: test cubes{suffix}', color=INK)
        fig.tight_layout()

        coverage_fig, ax = plt.subplots(figsize=(7, 3.8))
        _style(ax)
        positions = np.arange(len(labels))
        ax.bar(positions - 0.19, summary['coverage_68'], width=0.36, label='68 % interval', color=SERIES[0])
        ax.bar(positions + 0.19, summary['coverage_95'], width=0.36, label='95 % interval', color=SERIES[1])
        ax.axhline(0.68, color=SERIES[0], linestyle='--', linewidth=1)
        ax.axhline(0.95, color=SERIES[1], linestyle='--', linewidth=1)
        ax.set_xticks(positions, labels)
        ax.set(ylim=(0, 1), ylabel='Fraction containing the true value')
        ax.legend(frameon=False, loc='lower right')
        coverage_fig.tight_layout()
        return fig, coverage_fig

    def calibration(self, evaluation, title='Calibration'):
        '''Empirical vs nominal coverage of central intervals, one panel per parameter.'''
        curve = evaluation.coverage_curve()
        labels = _labels(evaluation.parameter_names, self.labels)
        fig, axes = _grid(len(labels), 3.3, 3.1, ncols=min(5, len(labels)))
        for ax, name, label in zip(axes, evaluation.parameter_names, labels):
            c = curve[curve.param == name]
            ax.fill_between(c.level, c.ci_low, c.ci_high, color=BAND, linewidth=0)
            ax.plot([0, 1], [0, 1], color=MUTED, linestyle='--', linewidth=1)
            ax.plot(c.level, c.empirical, color=SERIES[0], linewidth=2, marker='o', ms=3)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Nominal coverage', title=label)
        axes[0].set_ylabel('Empirical coverage')
        fig.suptitle(f'{title} (band: 95 % binomial interval)', color=INK)
        fig.tight_layout()
        return fig

    def sbc(self, evaluation, bins=20, title='SBC ranks'):
        '''Histogram of PIT values; flat within the band means calibrated marginals.'''
        from scipy import stats
        labels = _labels(evaluation.parameter_names, self.labels)
        n = len(evaluation.pit)
        low, high = stats.binom.ppf([0.005, 0.995], n, 1 / bins)
        fig, axes = _grid(len(labels), 3.3, 3.0, ncols=min(5, len(labels)))
        for j, (ax, label) in enumerate(zip(axes, labels)):
            ax.axhspan(low, high, color=BAND, linewidth=0)
            ax.axhline(n / bins, color=MUTED, linewidth=1)
            ax.hist(evaluation.pit[:, j], bins=bins, range=(0, 1), color=SERIES[0], edgecolor='#fcfcfb',
                    linewidth=1)
            ax.set(xlim=(0, 1), xlabel='P(θ < θ$_{true}$ | x)', title=label)
        axes[0].set_ylabel('Count')
        fig.suptitle(f'{title} (band: 99 % for uniform ranks)', color=INK)
        fig.tight_layout()
        return fig

    def tarp(self, evaluation, title='TARP'):
        import matplotlib.pyplot as plt
        alpha, ecp = evaluation.tarp_curve()
        fig, ax = plt.subplots(figsize=(4.2, 4))
        _style(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle='--', linewidth=1)
        ax.plot(alpha, ecp, color=SERIES[0], linewidth=2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Credibility level α', ylabel='Expected coverage',
               title=f'{title}: max |ECP − α| = {np.abs(ecp - alpha).max():.3f}')
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ posteriors
    def _corner(self, samples, names, truth=None, fisher=None, title='', fiducial=None):
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter
        labels = _labels(names, self.labels)
        p = len(names)
        fig, axes = plt.subplots(p, p, figsize=(1.9 * p + 1, 1.9 * p + 1), squeeze=False)
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
                _style(ax)
                ax.grid(False)
                if i == j:
                    ax.hist(samples[:, j], bins=50, range=ranges[j], density=True, histtype='step',
                            color=SERIES[0], linewidth=1.5)
                    if fisher is not None:
                        grid = np.linspace(*ranges[j], 200)
                        s = np.sqrt(fisher[1][j, j])
                        ax.plot(grid, np.exp(-0.5 * ((grid - fisher[0][j]) / s) ** 2) / (s * np.sqrt(2 * np.pi)),
                                color=SERIES[1], linewidth=1.5)
                    ax.set_yticks([])
                else:
                    hist, xe, ye = np.histogram2d(samples[:, j], samples[:, i], bins=40, range=[ranges[j], ranges[i]])
                    hist = gaussian_filter(hist, 1.0)
                    levels = _density_levels(hist)
                    if len(levels) == 2 and levels[0] < levels[1]:
                        ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), hist.T, levels=levels,
                                   colors=[SERIES[0]], linewidths=[1, 1.6])
                    if fisher is not None:
                        cov = fisher[1][np.ix_([j, i], [j, i])]
                        for d2, lw in ((2.30, 1.6), (6.18, 1.0)):
                            _ellipse(ax, (fisher[0][j], fisher[0][i]), cov, d2, color=SERIES[1], linewidth=lw)
                    ax.set_xlim(ranges[j])
                    ax.set_ylim(ranges[i])
                if truth is not None:
                    ax.axvline(truth[j], color=INK, linewidth=0.8, linestyle=':')
                    if i != j:
                        ax.axhline(truth[i], color=INK, linewidth=0.8, linestyle=':')
                if i == p - 1:
                    ax.set_xlabel(labels[j])
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i])
                elif j > 0:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=7)
        if fisher is not None:
            from matplotlib.lines import Line2D
            fig.legend([Line2D([], [], color=SERIES[0]), Line2D([], [], color=SERIES[1]),
                        Line2D([], [], color=INK, linestyle=':')],
                       ['NPE posterior', 'Fisher (68/95 %)', 'truth'], loc='upper right', frameon=False)
        fig.suptitle(title, color=INK)
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
        return self._corner(samples, self._names(result), truth=result.evaluation.truth[idx],
                            title=f'{result.name}: test cube {cube}')

    def fiducial_posterior(self, result, x, n_samples: int = 50_000, seed: int = 123):
        '''NPE posterior of one fiducial realization against the Fisher ellipses.'''
        names = self._names(result)
        samples = result.fit.sample(x, n_samples, seed).numpy()
        truth = result.fiducial.truth[0] if result.fiducial is not None else None
        fisher = None
        if result.fisher is not None and set(names) <= set(result.fisher.parameters):
            index = [result.fisher.parameters.index(n) for n in names]
            fisher = (np.array([result.fisher.fiducial[n] for n in names]), _marginal_cov(result.fisher, index))
        return self._corner(samples, names, truth=truth, fisher=fisher,
                            title=f'{result.name}: fiducial realization')

    def fiducial_scatter(self, result):
        '''Posterior means and stds over fiducial realizations (bias and spread at fixed cosmology).'''
        ev = result.fiducial
        labels = _labels(ev.parameter_names, self.labels)
        fig, axes = _grid(len(labels), 3.3, 3.0, ncols=min(5, len(labels)))
        for j, (ax, label) in enumerate(zip(axes, labels)):
            z = (ev.means[:, j] - ev.truth[:, j]) / ev.stds[:, j]
            ax.hist(z, bins=30, range=(-4, 4), density=True, color=SERIES[0], edgecolor='#fcfcfb', linewidth=1)
            grid = np.linspace(-4, 4, 200)
            ax.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color=MUTED, linewidth=1)
            ax.set(xlabel='(mean − fiducial) / std', title=f'{label}: ⟨z⟩ = {z.mean():.2f}')
        fig.suptitle(f'{result.name}: fiducial realizations (curve: unit normal)', color=INK)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ comparisons
    def compare_errors(self, results: Mapping, parameters=PARAMETER_NAMES):
        '''Per-parameter 1-sigma errors across experiments: NPE at the fiducial vs Fisher (GIGANTES Fig. 1-like).'''
        names = list(results)
        labels = _labels(parameters, self.labels)
        fig, axes = _grid(len(parameters), 3.6, 0.32 * len(names) + 1.6, ncols=min(5, len(parameters)))
        y = np.arange(len(names))
        for j, (ax, parameter, label) in enumerate(zip(axes, parameters, labels)):
            npe, fisher = [], []
            for name in names:
                r = results[name]
                npe.append(r.fiducial.stds[:, r.fiducial.parameter_names.index(parameter)].mean()
                           if r.fiducial is not None and parameter in r.fiducial.parameter_names else np.nan)
                fisher.append(dict(zip(r.fisher.parameters, r.fisher.sigma)).get(parameter, np.nan)
                              if r.fisher is not None else np.nan)
            ax.barh(y - 0.19, npe, height=0.36, color=SERIES[0], label='NPE ⟨σ⟩ at fiducial')
            ax.barh(y + 0.19, fisher, height=0.36, color=SERIES[1], label='Fisher σ')
            ax.set_xscale('log')
            ax.set_yticks(y, names if j % axes[0].get_gridspec().ncols == 0 else [''] * len(names))
            ax.invert_yaxis()
            ax.set_title(label)
            ax.grid(axis='y', visible=False)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper center', ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig

    def compare_metrics(self, comparison, metric='RMSE'):
        '''One panel per parameter: a test-set metric for every experiment.'''
        params = list(dict.fromkeys(comparison.param))
        labels = _labels(params, self.labels)
        experiments = list(dict.fromkeys(comparison.experiment))
        methods = list(dict.fromkeys(comparison.method))
        if len(methods) > len(SERIES):
            raise ValueError('Too many methods to colour distinctly.')
        fig, axes = _grid(len(params), 3.6, 0.32 * len(experiments) + 1.6, ncols=min(5, len(params)))
        y = np.arange(len(experiments))
        height = 0.8 / len(methods)
        for j, (ax, param, label) in enumerate(zip(axes, params, labels)):
            for m, method in enumerate(methods):
                rows = comparison[(comparison.param == param) & (comparison.method == method)].set_index('experiment')
                values = [rows[metric].get(e, np.nan) for e in experiments]
                ax.barh(y + (m - (len(methods) - 1) / 2) * height, values, height=height * 0.9,
                        color=SERIES[m], label=method.upper())
            if metric.startswith('coverage_'):
                ax.axvline(float(metric.split('_')[1]) / 100, color=MUTED, linestyle='--', linewidth=1)
            ax.set_yticks(y, experiments if j % axes[0].get_gridspec().ncols == 0 else [''] * len(experiments))
            ax.invert_yaxis()
            ax.set(title=label, xlabel=metric)
            ax.grid(axis='y', visible=False)
        if len(methods) > 1:
            handles, legend_labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='upper center', ncol=len(methods), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.94 if len(methods) > 1 else 1))
        return fig

    # ------------------------------------------------------------------ Fisher
    def fisher_ellipses(self, fishers: Mapping, parameters=None):
        '''Triangle of 68 % Fisher ellipses for up to 8 experiments.'''
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        names = list(fishers)
        if len(names) > len(SERIES):
            raise ValueError(f'At most {len(SERIES)} experiments per figure (got {len(names)}); split them.')
        parameters = tuple(parameters or next(iter(fishers.values())).parameters)
        labels = _labels(parameters, self.labels)
        p = len(parameters)
        fig, axes = plt.subplots(p, p, figsize=(1.9 * p + 1.5, 1.9 * p + 1), squeeze=False)
        covs = {}
        for name, f in fishers.items():
            index = [f.parameters.index(q) for q in parameters]
            covs[name] = (np.array([f.fiducial[q] for q in parameters]), _marginal_cov(f, index))
        span = np.max([np.sqrt(np.diag(c)) for _, c in covs.values()], axis=0) * 3.2
        center = next(iter(covs.values()))[0]
        for i in range(p):
            for j in range(p):
                ax = axes[i, j]
                if j > i:
                    ax.set_visible(False)
                    continue
                _style(ax)
                ax.grid(False)
                for k, (name, (mean, cov)) in enumerate(covs.items()):
                    if i == j:
                        grid = np.linspace(center[j] - span[j], center[j] + span[j], 300)
                        s = np.sqrt(cov[j, j])
                        ax.plot(grid, np.exp(-0.5 * ((grid - mean[j]) / s) ** 2), color=SERIES[k], linewidth=1.5)
                    else:
                        _ellipse(ax, (mean[j], mean[i]), cov[np.ix_([j, i], [j, i])], 2.30, color=SERIES[k],
                                 linewidth=1.5)
                ax.set_xlim(center[j] - span[j], center[j] + span[j])
                if i != j:
                    ax.set_ylim(center[i] - span[i], center[i] + span[i])
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
                ax.tick_params(labelsize=7)
        fig.legend([Line2D([], [], color=SERIES[k]) for k in range(len(names))], names, loc='upper right',
                   frameon=False)
        fig.suptitle('Fisher forecast, 68 % (1 (Gpc/h)$^3$, z = 0.5)', color=INK)
        fig.tight_layout()
        return fig

    def correlation(self, fisher, title=None):
        '''Correlation matrix of the features at the fiducial cosmology (GIGANTES Fig. 7-like).'''
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('div', ['#104281', '#3987e5', '#f0efec', '#e66767', '#a1201f'])
        corr = fisher.correlation()
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        image = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
        sizes = _block_sizes(fisher)
        edges = np.cumsum([0] + sizes)
        for e in edges[1:-1]:
            ax.axhline(e - 0.5, color=INK, linewidth=0.6)
            ax.axvline(e - 0.5, color=INK, linewidth=0.6)
        centers = 0.5 * (edges[1:] + edges[:-1]) - 0.5
        ax.set_xticks(centers, fisher.blocks, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(centers, fisher.blocks, fontsize=8)
        fig.colorbar(image, ax=ax, label='Pearson correlation', shrink=0.85)
        ax.set_title(title or 'Feature correlations, fiducial', color=INK)
        fig.tight_layout()
        return fig

    def convergence(self, table, title='Fisher convergence'):
        names = list(dict.fromkeys(table.param))
        fig, axes = _grid(2, 4.3, 3.4, ncols=2)
        for ax, kind in zip(axes, ('covariance', 'derivative')):
            t = table[table.kind == kind]
            for k, name in enumerate(names):
                rows = t[t.param == name]
                ax.plot(rows.n, rows.ratio_to_full, marker='o', ms=4, linewidth=2, color=SERIES[k],
                        label=ALL_PARAMETER_LABELS.get(name, name))
            ax.axhline(1, color=MUTED, linewidth=1)
            ax.set(xlabel=f'Realizations used for the {kind}', ylabel='σ / σ(all)', title=kind)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(title, color=INK)
        fig.tight_layout()
        return fig


def _marginal_cov(fisher, index):
    return np.linalg.inv(fisher.fisher)[np.ix_(index, index)]


def _block_sizes(fisher):
    sizes = getattr(fisher, 'block_sizes', None)
    if sizes:
        return list(sizes)
    return [fisher.mean.size]


__all__ = ['SBIPlotter', 'SERIES', 'PARAMETER_LABELS']
