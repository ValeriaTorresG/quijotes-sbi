from __future__ import annotations

import argparse
from pathlib import Path

from .config import ENVIRONMENT_FILES, EvaluationConfig, SpectrumConfig, TrainingConfig
from .data import QuijoteDataLoader
from .evaluation import PosteriorEvaluator
from .inference import SBITrainer
from .pipeline import SBIPipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path('/pscratch/sd/v/vtorresg/quijotes/PowerSpectrum/FoF/latin_hypercube'))
    parser.add_argument('--params-file', type=Path, default=Path(__file__).resolve().parents[1] / 'latin_hypercube_params.txt')
    parser.add_argument('--envs', nargs='+', choices=tuple(ENVIRONMENT_FILES), default=['void'])
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--n-cubes', type=int, default=2000)
    parser.add_argument('--n-train', type=int, default=1800)
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--kmin', type=float, default=0.01)
    parser.add_argument('--kmax', type=float, default=0.5)
    parser.add_argument('--transform', choices=('log10', 'none'), default='log10')
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--train-seed', type=int, default=42)
    parser.add_argument('--sample-seed', type=int, default=123)
    parser.add_argument('--hidden-features', type=int, default=128)
    parser.add_argument('--num-transforms', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args(argv)

    try:
        spectrum = SpectrumConfig(kmin=args.kmin, kmax=args.kmax, transform=args.transform)
        training = TrainingConfig(n_train=args.n_train, split_seed=args.split_seed, train_seed=args.train_seed,
                                  hidden_features=args.hidden_features, num_transforms=args.num_transforms,
                                  batch_size=args.batch_size, learning_rate=args.learning_rate,
                                  validation_fraction=args.validation_fraction, patience=args.patience,
                                  max_epochs=args.max_epochs, device=args.device)
        evaluation = EvaluationConfig(n_samples=args.n_samples, sample_seed=args.sample_seed)
        n_validation = int(args.n_train * args.validation_fraction)
    except ValueError as error:
        parser.error(str(error))

    name = '_'.join(args.envs)
    output = args.output or Path('results') / ('comparison' if args.compare else name)
    loader = QuijoteDataLoader(args.base, args.params_file, config=spectrum)
    pipeline = SBIPipeline(loader, SBITrainer(training), PosteriorEvaluator(evaluation))

    if args.compare:
        experiments = {env: env for env in args.envs}
        if len(args.envs) > 1:
            experiments['combined'] = args.envs
    else:
        experiments = {name: args.envs}

    print(f'tries: {list(experiments)} | cubes: {args.n_cubes} | Train: {args.n_train}', flush=True)

    results = pipeline.run_many(experiments, range(args.n_cubes))
    for label, result in results.items():
        destination = output / label if args.compare else output
        result.save_diagnostics(destination)

        if args.plots:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from .plotting import SBIPlotter

            plotter = SBIPlotter()
            figures = (*plotter.diagnostics(result), plotter.posterior(result, n_samples=args.n_samples, seed=args.sample_seed))
            for filename, fig in zip(('prediction.png', 'coverage.png', 'posterior.png'), figures):
                fig.savefig(destination / filename, dpi=200, bbox_inches='tight')
                plt.close(fig)

        print(f'\n{label}\n{result.summary.to_string(index=False)}')
        print(f'---> results: {destination.resolve()}', flush=True)

    if args.compare:
        pipeline.compare(results).to_csv(output / 'comparison.csv', index=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())