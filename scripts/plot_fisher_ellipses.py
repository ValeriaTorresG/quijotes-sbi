'''Overlay the NPE posteriors and Fisher ellipses of several saved experiments.

Each experiment gets one colour: filled 68/95 % contours of its NPE posterior
for one fiducial realization (the same one for every model) and its dashed
68 % Fisher ellipse. Uses the saved model.pt and the Fisher matrix stored in
diagnostics.npz (runs made with ``train --fisher``); nothing is retrained.

    python scripts/plot_fisher_ellipses.py results/ambientes \
        all='All halos' void='Void halos' random_void='Random voids' envs_rv_all='Combined'

Order matters: list the widest posteriors first so the tight ones stay on top.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path, help='Folder with one subfolder per experiment')
    parser.add_argument('experiments', nargs='+', metavar='NAME[=LABEL]')
    parser.add_argument('--no-posterior', action='store_true', help='Only the Fisher ellipses')
    parser.add_argument('--realization', type=int, default=None,
                        help='Fiducial realization (default: the first one used in the fiducial check)')
    parser.add_argument('--params', nargs='+', default=None,
                        help='Subset of parameters to show, e.g. Omega_m sigma8 n_s')
    parser.add_argument('--n-samples', type=int, default=50_000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    plotter = SBIPlotter()
    experiments = {}
    for item in args.experiments:
        name, _, label = item.partition('=')
        experiment = load_experiment(args.run / name)
        if experiment.fisher is None:
            raise SystemExit(f'{name}: no Fisher matrix saved (train with --fisher).')
        experiments[label or name] = experiment

    if args.no_posterior:
        fig = plotter.fisher_ellipses({label: e.fisher for label, e in experiments.items()}, parameters=args.params)
        default = 'fisher_ellipses_selected.png'
    else:
        first = next(iter(experiments.values()))
        realization = args.realization
        if realization is None:
            realization = int(first.fiducial.cube_ids[0])
        features = _Features()
        entries = {}
        for label, experiment in experiments.items():
            print(f'sampling {experiment.name} at fiducial realization {realization}', flush=True)
            samples = sample_fiducial(experiment, realization, args.n_samples, args.seed, args.device, features)
            entries[label] = (samples, experiment.fisher)
        names = first.evaluation.parameter_names
        if args.params:
            unknown = set(args.params) - set(names)
            if unknown:
                raise SystemExit(f'Unknown parameters {sorted(unknown)}; choose from {names}.')
            columns = [names.index(p) for p in args.params]
            entries = {k: (s[:, columns], f) for k, (s, f) in entries.items()}
            names = tuple(args.params)
        truth = np.array([first.fisher.fiducial[n] for n in names])
        fig = plotter.posterior_comparison(entries, names, truth=truth,
                                           title=f'Fiducial realization {realization}')
        default = 'posteriors_fisher_selected.png'
    output = args.output or args.run / default
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(output.resolve())


if __name__ == '__main__':
    main()
