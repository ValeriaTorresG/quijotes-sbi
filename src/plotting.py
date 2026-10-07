from __future__ import annotations

import numpy as np

from .config import PARAMETER_LABELS, PARAMETER_NAMES
from .pipeline import ExperimentResult


class SBIPlotter:
    def __init__(self, labels=None):
        self.labels = None if labels is None else tuple(labels)

    def _labels(self, result: ExperimentResult):
        names = result.evaluation.parameter_names
        labels = self.labels
        if labels is None:
            known = dict(zip(PARAMETER_NAMES, PARAMETER_LABELS))
            labels = tuple(known.get(name, name) for name in names)
        if len(labels) != len(names):
            raise ValueError('labels must match the number of parameters.')
        return labels

    @staticmethod
    def _grid(n_parameters):
        import matplotlib.pyplot as plt

        ncols = min(3, n_parameters)
        nrows = (n_parameters + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.5 * nrows), squeeze=False)
        for ax in axes.ravel()[n_parameters:]:
            ax.set_visible(False)
        return fig, axes

    def diagnostics(self, result: ExperimentResult):
        '''Return (prediction_figure, coverage_figure); caller can show/save them.'''
        import matplotlib.pyplot as plt

        ev = result.evaluation
        labels = self._labels(result)
        prediction_fig, axes = self._grid(len(labels))

        for j, label in enumerate(labels):
            ax = axes.ravel()[j]
            ax.vlines(ev.truth[:, j], ev.low68[:, j], ev.high68[:, j], color='royalblue', alpha=0.25)
            ax.scatter(ev.truth[:, j], ev.means[:, j], s=12, color='royalblue', alpha=0.7)
            lower = min(ev.truth[:, j].min(), ev.low68[:, j].min(), ev.means[:, j].min())
            upper = max(ev.truth[:, j].max(), ev.high68[:, j].max(), ev.means[:, j].max())
            ax.plot([lower, upper], [lower, upper], 'k--')
            ax.set(xlabel='True value', ylabel='Mean posterior', title=label)
            ax.grid(alpha=0.2)

        prediction_fig.suptitle(' + '.join(result.fit.split.data.envs))
        prediction_fig.tight_layout()

        coverage_fig, ax = plt.subplots(figsize=(7, 4))
        summary, positions = ev.summary, np.arange(len(labels))
        ax.bar(positions - 0.18, summary['coverage_68'], width=0.36, label='68\% interval', color='royalblue')
        ax.bar(positions + 0.18, summary['coverage_95'], width=0.36, label='95\% interval', color='darkorange')
        ax.axhline(0.68, color='royalblue', linestyle='--')
        ax.axhline(0.95, color='darkorange', linestyle='--')

        ax.set_xticks(positions, labels)
        ax.set(ylim=(0, 1), ylabel='Fraction containing the true value')
        ax.legend()
        coverage_fig.tight_layout()
        return prediction_fig, coverage_fig

    def posterior(self, result: ExperimentResult, idx: int = 0, n_samples: int = 50_000, seed: int = 123):
        '''Return marginal posterior histograms for one held-out cube.'''
        if not 0 <= idx < len(result.fit.split.test_rows):
            raise IndexError('idx must index a held-out cube.')

        labels = self._labels(result)
        samples = result.fit.sample(result.fit.split.x_test[idx], n_samples, seed).numpy()
        truth, means = result.evaluation.truth[idx], samples.mean(axis=0)

        fig, axes = self._grid(len(labels))
        for j, label in enumerate(labels):
            ax = axes.ravel()[j]
            ax.hist(samples[:, j], bins=60, density=True, color='royalblue', alpha=0.65)
            ax.axvline(truth[j], color='black', linestyle='--', label=f'True: {truth[j]:.4f}')
            ax.axvline(means[j], color='darkorange', label=f'Mean posterior: {means[j]:.4f}')
            ax.set(xlabel=label, ylabel='PDF')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.2)
        cube_id = result.fit.split.test_cube_ids[idx]
        fig.suptitle(f'{" + ".join(result.fit.split.data.envs)} - cube {cube_id}')
        fig.tight_layout()
        return fig
