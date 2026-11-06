'''Relative improvement of the constraints with respect to a reference experiment.

For each parameter i and experiment X:

    R_i^method(X) = sigma_i^method(X) / sigma_i^method(reference)

with method = Fisher (blue) and NPE (red). R < 1 means X constrains the
parameter better than the reference (by default ``all``, the full halo P(k)).
Normalizing each method by its own reference removes the absolute scale
differences between Fisher and NPE (prior, Gaussian approximation, ...).

    python scripts/plot_relative_constraints.py results/ambientes
    python scripts/plot_relative_constraints.py results/ambientes --fisher both

Fisher estimators:
  standard    F = dmu C^-1 dmu (saved in fisher.csv). Optimistic when the
              derivatives are noisy (Omega_b, h with long data vectors).
  combined    geometric mean of the standard and the compressed (conservative)
              Fisher, Coulton & Wandelt (2023). Recomputed here from the cache
              with the compressed part averaged over many random half-splits;
              runs saved before that change stored a single, noisy split.
  both        standard (solid) and combined (dashed).

NPE sigma: mean posterior std over the fiducial realizations evaluated in the
run (``--npe fiducial``, same cosmology as Fisher) or over the held-out LH cubes
(``--npe test``). For prior-dominated parameters (h, Omega_b) the NPE width is
capped by the prior, which pushes its ratios towards 1.
'''
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import ALL_PARAMETER_LABELS, PARAMETER_NAMES  # noqa: E402
from src.plotting import REFERENCE, SBIPlotter, tex  # noqa: E402

SHORT_LABELS = {'all': 'All', 'envs': 'Envs', 'envs_rv': 'Envs\n+ Rand voids', 'envs_rv_all': 'Envs + Rand voids\n+ All',
                'void': 'Void', 'sheet': 'Sheet', 'filament': 'Filament', 'knot': 'Knot',
                'random_void': 'Rand voids'}
REFERENCE_SYMBOL = {'all': r'{\rm All}', 'envs': r'{\rm Envs}'}
REFERENCE_TEXT = {'all': 'all-halo power-spectrum', 'envs': 'environment power-spectra'}
FISHER_COLOR, NPE_COLOR = 'royalblue', 'crimson'


def recomputed_fisher(run: Path, experiment: str, cache: dict):
    '''Fisher of a saved experiment, recomputed from the cache with its own feature configuration.'''
    from src.fisher import FisherForecaster
    from src.replot import _Features, feature_config
    metadata = json.loads((run / experiment / 'metadata.json').read_text())
    config = feature_config(metadata)
    if config is None:
        raise SystemExit(f'{run / experiment}: older run without feature configuration; use --fisher standard.')
    features = cache.setdefault('features', _Features())
    loader = features.loader(config, metadata.get('data'))
    key = id(loader)
    if key not in cache:
        cache[key] = FisherForecaster(loader)
    result = cache[key].compute(','.join(metadata['blocks']))
    saved = pd.read_csv(run / experiment / 'fisher.csv').set_index('param').sigma_fisher
    if not np.allclose(result.sigma, saved.loc[list(result.parameters)].to_numpy(), rtol=1e-6):
        raise SystemExit(f'{experiment}: recomputed standard Fisher differs from the saved one; features changed?')
    return pd.DataFrame({'standard': result.sigma, 'combined': result.sigma_combined,
                         'compressed': result.sigma_compressed}, index=list(result.parameters))


def collect(run: Path, experiments, fisher: str, npe: str) -> pd.DataFrame:
    cache, rows = {}, []
    for e in experiments:
        folder = run / e
        npe_file = folder / ('fiducial_summary.csv' if npe == 'fiducial' else 'summary.csv')
        if not npe_file.is_file():
            raise SystemExit(f'{npe_file} not found (fiducial check off?); use --npe test.')
        npe_table = pd.read_csv(npe_file).set_index('param')
        if fisher == 'standard':
            fisher_table = pd.read_csv(folder / 'fisher.csv').set_index('param').rename(
                columns={'sigma_fisher': 'standard'})[['standard']]
        else:
            print(f'recomputing Fisher for {e}', flush=True)
            fisher_table = recomputed_fisher(run, e, cache)
        for p in npe_table.index:
            row = dict(experiment=e, param=p, sigma_npe=npe_table.loc[p, 'mean_posterior_std'],
                       n_npe=int(npe_table.loc[p, 'n']) if 'n' in npe_table else np.nan)
            for column in fisher_table.columns:
                row[f'sigma_fisher_{column}'] = fisher_table.loc[p, column]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path, help='Folder with one subfolder per experiment')
    parser.add_argument('--experiments', nargs='+', default=['all', 'envs', 'envs_rv', 'envs_rv_all'],
                        help='Order on the x axis')
    parser.add_argument('--reference', default='all', help='Experiment that defines R = 1')
    parser.add_argument('--labels', nargs='+', default=None, help='x tick labels, one per experiment')
    parser.add_argument('--fisher', choices=('standard', 'combined', 'both'), default='standard')
    parser.add_argument('--npe', choices=('fiducial', 'test'), default='fiducial')
    parser.add_argument('--params', nargs='+', default=list(PARAMETER_NAMES))
    parser.add_argument('--title', default='Relative cosmological constraints: Fisher vs. NPE')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.experiments):
        raise SystemExit('--labels needs one label per experiment.')
    names = list(dict.fromkeys(args.experiments + [args.reference]))
    table = collect(args.run, names, args.fisher, args.npe)
    sigma_columns = [c for c in table if c.startswith('sigma_')]
    reference = table[table.experiment == args.reference].set_index('param')[sigma_columns]
    ratios = table[table.experiment.isin(args.experiments)].copy()
    for column in sigma_columns:
        ratios['R_' + column[len('sigma_'):]] = ratios[column].to_numpy() / reference.loc[ratios.param, column].to_numpy()

    SBIPlotter()  # LaTeX style
    labels = args.labels or [SHORT_LABELS.get(e, tex(e)) for e in args.experiments]
    reference_symbol = REFERENCE_SYMBOL.get(args.reference, r'{\rm %s}' % tex(args.reference))
    fisher_lines = {'standard': [('standard', '-', 'Fisher')],
                    'combined': [('combined', '-', 'Fisher (combined)')],
                    'both': [('standard', '-', 'Fisher (standard)'), ('combined', '--', 'Fisher (combined)')]}[args.fisher]
    x = np.arange(len(args.experiments))
    ncols = 3
    nrows = (len(args.params) + 1 + ncols - 1) // ncols  # one extra slot for the legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False, sharey=True)
    axes = axes.ravel()
    top = 1.15
    for k, (ax, p) in enumerate(zip(axes, args.params)):
        rows = ratios[ratios.param == p].set_index('experiment').loc[args.experiments]
        ax.axhline(1, color=REFERENCE, ls='--', lw=1)
        for column, style, _ in fisher_lines:
            ax.plot(x, rows[f'R_fisher_{column}'], marker='o', ls=style, color=FISHER_COLOR)
        ax.plot(x, rows.R_npe, marker='s', color=NPE_COLOR)
        top = max(top, 1.08 * np.nanmax(rows[[c for c in rows if c.startswith('R_')]].to_numpy()))
        ax.set_xticks(x, labels, fontsize=9, multialignment='center')
        ax.set_xlim(-0.45, len(x) - 0.55)
        ax.set_title(ALL_PARAMETER_LABELS.get(p, tex(p)))
        if k % ncols == 0:
            ax.set_ylabel(r'Relative uncertainty $\sigma / \sigma_{%s}$' % reference_symbol)
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].set_ylim(0, top)
    legend_ax = axes[len(args.params)]
    legend_ax.axis('off')
    handles = [Line2D([], [], color=FISHER_COLOR, marker='o', ls=style) for _, style, _ in fisher_lines]
    handles += [Line2D([], [], color=NPE_COLOR, marker='s'), Line2D([], [], color=REFERENCE, ls='--')]
    legend_labels = [label for _, _, label in fisher_lines] + ['NPE', f'Reference ({SHORT_LABELS.get(args.reference, tex(args.reference))})']
    legend_ax.legend(handles, legend_labels, loc='center', fontsize=12)
    for ax in axes[len(args.params) + 1:]:
        fig.delaxes(ax)
    if args.title:
        fig.suptitle(tex(args.title))
    fig.tight_layout()

    output = args.output or args.run / f'relative_constraints_{args.fisher}_{args.npe}.png'
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    ratios.to_csv(output.with_suffix('.csv'), index=False)

    n_npe = int(table.n_npe.dropna().iloc[0]) if table.n_npe.notna().any() else None
    estimator = {'standard': 'standard Fisher forecasts',
                 'combined': 'Fisher forecasts with the combined estimator of Coulton & Wandelt (2023)',
                 'both': 'standard (solid) and combined (dashed; Coulton & Wandelt 2023) Fisher forecasts'}[args.fisher]
    where = (f'averaged over {n_npe} fiducial realizations' if args.npe == 'fiducial'
             else f'averaged over {n_npe} held-out latin-hypercube simulations')
    caption = (f'Relative marginalized parameter uncertainties from {estimator} and neural posterior estimation '
               f'(NPE). Each method is normalized to its corresponding '
               f'{REFERENCE_TEXT.get(args.reference, args.reference)} constraint. NPE uncertainties are posterior standard deviations {where}.')
    output.with_suffix('.caption.txt').write_text(caption + '\n')
    columns = [c for c in ratios if c.startswith('R_')]
    print(ratios.pivot(index='param', columns='experiment', values=columns)
          .reindex(args.params).reindex(columns=args.experiments, level=1).round(3).to_string())
    print(f'\n{caption}\n{output.resolve()}')


if __name__ == '__main__':
    main()
