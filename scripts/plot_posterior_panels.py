'''Compare posterior shapes: one panel per observable, one row per parameter pair.

SBI (solid blue contour, faint fill) vs Fisher (dashed red ellipse) for one
fiducial realization, with identical axis limits across each row. Uses the
saved model.pt and Fisher matrix of each experiment; nothing is retrained.

    python scripts/plot_posterior_panels.py results/ambientes
    python scripts/plot_posterior_panels.py results/ambientes \
        all='All halos' envs='Environments' envs_rv='Envs + random voids' envs_rv_all='Full combination' \
        --pairs Omega_m,sigma8 n_s,sigma8 --levels 68
'''
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.plotting import SBIPlotter  # noqa: E402
from src.replot import _Features, load_experiment, sample_fiducial  # noqa: E402

DEFAULT = ['all=All halos', 'envs=Environments', 'envs_rv=Envs + random voids', 'envs_rv_all=Full combination']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path, help='Folder with one subfolder per experiment')
    parser.add_argument('experiments', nargs='*', default=DEFAULT, metavar='NAME[=TITLE]',
                        help='Columns, left to right (default: all, envs, envs_rv, envs_rv_all)')
    parser.add_argument('--pairs', nargs='+', default=['Omega_m,sigma8', 'n_s,sigma8'],
                        help="Rows as x,y parameter pairs, or 'all' for every pair of parameters")
    parser.add_argument('--clip-prior', action='store_true',
                        help='Limit the axes to the latin-hypercube prior (Fisher ellipses may be cut)')
    parser.add_argument('--levels', nargs='+', type=int, choices=(68, 95), default=[68])
    parser.add_argument('--realization', type=int, default=None,
                        help='Fiducial realization (default: the first one used in the fiducial check)')
    parser.add_argument('--n-samples', type=int, default=50_000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    if args.pairs != ['all']:
        pairs = [tuple(p.split(',')) for p in args.pairs]
        if any(len(p) != 2 for p in pairs):
            raise SystemExit("--pairs takes x,y pairs such as Omega_m,sigma8, or 'all'")
    plotter = SBIPlotter()
    features = _Features()
    entries, names, truth, realization = {}, None, None, args.realization
    for item in args.experiments:
        name, _, title = item.partition('=')
        experiment = load_experiment(args.run / name)
        if realization is None:
            realization = int(experiment.fiducial.cube_ids[0])
        print(f'sampling {name} at fiducial realization {realization}', flush=True)
        samples = sample_fiducial(experiment, realization, args.n_samples, args.seed, args.device, features)
        names = experiment.evaluation.parameter_names
        truth = np.array([experiment.fiducial.truth[0][i] for i in range(len(names))]) \
            if experiment.fiducial is not None else None
        entries[title or name] = (samples, experiment.fisher)
    if args.pairs == ['all']:
        pairs = [(x, y) for i, y in enumerate(names) for x in names[:i]]  # corner-plot order
    bounds = None
    if args.clip_prior:
        from src.config import PriorConfig
        prior = PriorConfig()
        bounds = {p: (lo, hi) for p, lo, hi in zip(names, prior.low, prior.high)}
    fig = plotter.posterior_panels(entries, names, pairs=pairs, truth=truth,
                                   levels=tuple(lvl / 100 for lvl in args.levels), bounds=bounds)
    suffix = '_all_pairs' if args.pairs == ['all'] else ''
    output = args.output or args.run / f'posterior_panels{suffix}.png'
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(output.resolve())


if __name__ == '__main__':
    main()
