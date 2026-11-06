'''Command line: python -m src {inventory,validate,fisher,train} ...

``python -m src --envs void ...`` (no subcommand) is the old interface and runs ``train``.
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .config import ALL_SETS, PARAMETER_NAMES, DataLayout, EvaluationConfig, FeatureConfig, FisherConfig, \
    MomentTrainingConfig, SpectrumConfig, TrainingConfig, PK_SAMPLES
from .products import PRODUCTS, SAMPLES

COMMANDS = ('inventory', 'validate', 'fisher', 'train', 'replot')


def _common(parser):
    g = parser.add_argument_group('data')
    g.add_argument('--root', type=Path, default=DataLayout.root, help='Quijote data root')
    g.add_argument('--params-file', type=Path, default=DataLayout.params_file)
    g.add_argument('--cache-dir', type=Path, default=None, help='Cache of validated raw products (default ./cache)')
    g.add_argument('--workers', type=int, default=16, help='Threads for file reading')
    g.add_argument('--allow-missing', action='store_true',
                   help='Use the complete subset when realizations are missing (default: fail)')
    g.add_argument('--no-verify-cache', action='store_true', help='Skip the source-file fingerprint check')
    g.add_argument('--base', type=Path, default=None, help=argparse.SUPPRESS)  # legacy, ignored with a warning


def _features(parser):
    g = parser.add_argument_group('features')
    g.add_argument('--kmin', type=float, default=0.01)
    g.add_argument('--kmax', type=float, default=0.5)
    g.add_argument('--shot-noise', choices=('subtracted', 'raw'), default='subtracted',
                   help='Use the shot-noise-subtracted P(k) (default) or the raw one')
    g.add_argument('--transform', choices=('asinh', 'log10', 'none'), default='asinh',
                   help='P(k) transform: asinh(P/scale) accepts negative subtracted P; log10 needs --shot-noise raw')
    g.add_argument('--asinh-scale', type=float, default=1000.0, help='Scale s in asinh(P/s), (Mpc/h)^3')
    g.add_argument('--rsd-multipoles', type=int, nargs='+', default=[0, 2])
    g.add_argument('--astra-bins', type=int, default=10)
    g.add_argument('--hmf-edges', type=float, nargs='+', default=list(FeatureConfig.hmf_log10_mass_edges))


def _experiments(parser):
    g = parser.add_argument_group('experiments')
    g.add_argument('--experiment', '-e', nargs='+', default=[], metavar='NAME=BLOCKS',
                   help="e.g. void=pk:void  envs=pk:envs  'full=pk:envs,nbar:envs,hmf:envs,astra:envs'")
    g.add_argument('--preset', choices=('environments', 'full'), default=None)
    g.add_argument('--envs', nargs='+', choices=PK_SAMPLES, default=None,
                   help='Old interface: P(k) of these samples (concatenated, or each with --compare)')
    g.add_argument('--compare', action='store_true', help='With --envs: one model per sample plus the combination')


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m src', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('inventory', help='Completeness of every set/product (no reading)')
    _common(p)
    p.add_argument('--sets', nargs='+', choices=ALL_SETS, default=list(ALL_SETS))
    p.add_argument('--products', nargs='+', choices=PRODUCTS, default=list(PRODUCTS))
    p.add_argument('--output', type=Path, default=Path('results/inventory.csv'))

    p = sub.add_parser('validate', help='Read, validate and cache products; cross-check counts')
    _common(p)
    p.add_argument('--sets', nargs='+', choices=ALL_SETS, default=list(ALL_SETS))
    p.add_argument('--products', nargs='+', choices=PRODUCTS, default=['pk', 'astra', 'hmf'])
    p.add_argument('--output', type=Path, default=Path('results/validation'))

    p = sub.add_parser('fisher', help='Fisher forecasts (GIGANTES-style) for each experiment')
    _common(p), _features(p), _experiments(p)
    p.add_argument('--fisher-params', nargs='+', choices=PARAMETER_NAMES + ('Mnu',), default=list(PARAMETER_NAMES))
    p.add_argument('--n-covariance', type=int, default=None)
    p.add_argument('--n-derivative', type=int, default=None)
    p.add_argument('--convergence', action='store_true')
    p.add_argument('--output', type=Path, default=Path('results/fisher'))
    p.add_argument('--plots', action='store_true')

    p = sub.add_parser('replot', help='Redraw figures from saved results (no retraining)')
    p.add_argument('paths', nargs='+', type=Path, help='Result folders (searched recursively)')
    p.add_argument('--no-model', action='store_true', help='Skip figures that sample the saved model')
    p.add_argument('--n-samples', type=int, default=20_000, help='Posterior samples for corner plots')
    p.add_argument('--device', default='cpu')

    p = sub.add_parser('train', help='Train and validate NPE / Moment Networks')
    _common(p), _features(p), _experiments(p)
    g = p.add_argument_group('training')
    g.add_argument('--method', choices=('npe', 'mn', 'both'), default='npe')
    g.add_argument('--n-cubes', type=int, default=None, help='Use cube IDs 0..n-1 (default: all available)')
    g.add_argument('--n-train', type=int, default=None, help='Training cubes (default: 90 %%)')
    g.add_argument('--split-seed', type=int, default=42)
    g.add_argument('--train-seed', type=int, default=42)
    g.add_argument('--ensemble', type=int, default=5, help='NPE ensemble members')
    g.add_argument('--density-estimator', choices=('nsf', 'maf', 'mdn'), default='nsf')
    g.add_argument('--hidden-features', type=int, default=128)
    g.add_argument('--num-transforms', type=int, default=5)
    g.add_argument('--hidden-layers', type=int, default=2, help='Moment Network depth')
    g.add_argument('--embedding-dim', type=int, default=0, help='MLP embedding output size (0 = none)')
    g.add_argument('--compression', choices=('none', 'moped'), default='none')
    g.add_argument('--batch-size', type=int, default=128)
    g.add_argument('--learning-rate', type=float, default=5e-4)
    g.add_argument('--validation-fraction', type=float, default=0.1)
    g.add_argument('--patience', type=int, default=30)
    g.add_argument('--max-epochs', type=int, default=1000)
    g.add_argument('--device', default='auto')
    g = p.add_argument_group('validation')
    g.add_argument('--n-samples', type=int, default=5000)
    g.add_argument('--sample-seed', type=int, default=123)
    g.add_argument('--fiducial', type=int, default=500, help='Fiducial realizations for the fixed-cosmology check (0 = off)')
    g.add_argument('--fisher', action='store_true', help='Also compute Fisher errors for each experiment')
    g.add_argument('--fisher-params', nargs='+', choices=PARAMETER_NAMES + ('Mnu',), default=list(PARAMETER_NAMES))
    g.add_argument('--no-save-model', action='store_true')
    g.add_argument('--output', type=Path, default=None)
    g.add_argument('--plots', action='store_true')
    return parser


# ---------------------------------------------------------------------- helpers
def _store(args):
    from .store import ProductStore
    if args.base is not None:
        print('warning: --base is ignored; use --root (the layout below the root is fixed in DataLayout).',
              file=sys.stderr)
    layout = DataLayout(root=args.root, params_file=args.params_file)
    return ProductStore(layout, args.cache_dir, workers=args.workers, verify_cache=not args.no_verify_cache)


def _loader(args, store):
    from .features import QuijoteSummaryLoader
    spectrum = SpectrumConfig(kmin=args.kmin, kmax=args.kmax, transform=args.transform, shot_noise=args.shot_noise,
                              asinh_scale=args.asinh_scale)
    config = FeatureConfig(spectrum=spectrum, rsd_multipoles=tuple(args.rsd_multipoles), astra_bins=args.astra_bins,
                           hmf_log10_mass_edges=tuple(args.hmf_edges))
    return QuijoteSummaryLoader(store, config, allow_missing=args.allow_missing)


def _experiment_specs(args) -> dict[str, str]:
    from .features import PRESETS, parse_experiments
    specs: dict[str, str] = {}
    if args.preset:
        specs.update(PRESETS[args.preset])
    if args.envs:
        if args.compare:
            specs.update({env: f'pk:{env}' for env in args.envs})
            if len(args.envs) > 1:
                specs['combined'] = ','.join(f'pk:{e}' for e in args.envs)
        else:
            specs['_'.join(args.envs)] = ','.join(f'pk:{e}' for e in args.envs)
    for name, blocks in parse_experiments(args.experiment).items() if args.experiment else []:
        if name in specs:
            raise ValueError(f'Experiment {name!r} defined twice.')
        specs[name] = ','.join(b.name for b in blocks)
    if not specs:
        raise ValueError('Choose experiments with --experiment, --preset or --envs.')
    parse_experiments(f'{n}={s}' for n, s in specs.items())  # validate every spec up front
    return specs


def _save_figure(fig, path):
    import matplotlib.pyplot as plt
    fig.savefig(path, bbox_inches='tight')  # dpi from the plotting style
    plt.close(fig)


# ---------------------------------------------------------------------- commands
def cmd_inventory(args):
    store = _store(args)
    table = store.inventory(args.sets, args.products)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    import pandas as pd
    with pd.option_context('display.max_rows', 500, 'display.width', 160):
        print(table.to_string(index=False))
    print(f'---> {args.output.resolve()}')
    return 0


def cmd_validate(args):
    import pandas as pd
    from .features import check_consistency, Block
    store = _store(args)
    rows, problems = [], []
    for set_name in args.sets:
        tables = {}
        for product in args.products:
            for sample in SAMPLES[product]:
                try:
                    table = store.get(set_name, product, sample, allow_missing=args.allow_missing)
                except Exception as error:  # report and continue with the other products
                    problems.append(dict(set=set_name, product=product, sample=sample or '', error=str(error)))
                    print(f'FAIL {set_name}/{product}/{sample or ""}: {error}', flush=True)
                    continue
                rows.append(dict(set=set_name, product=product, sample=sample or '', n=len(table.ids)))
                print(f'ok   {set_name}/{product}/{sample or ""}: {len(table.ids)} realizations', flush=True)
                statistic = {'pk': 'pk', 'pk_rsd': 'pkrsd', 'astra': 'astra', 'hmf': 'hmf'}[product]
                tables[Block(statistic, sample or 'void')] = table
        pk_blocks = {b: t for b, t in tables.items() if b.statistic == 'pk'}
        for extra in [b for b in tables if b.statistic in ('astra', 'hmf')] or [None]:
            subset = dict(pk_blocks)
            if extra is not None:
                subset[extra] = tables[extra]
            if not subset:
                continue
            common = None
            for t in subset.values():
                common = t.ids if common is None else np.intersect1d(common, t.ids)
            try:
                check_consistency({b: t.take(common) for b, t in subset.items()})
            except Exception as error:
                problems.append(dict(set=set_name, product='cross-check', sample='', error=str(error)))
                print(f'FAIL {set_name} cross-check: {error}', flush=True)
        store._memory.clear()
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / 'validated.csv', index=False)
    pd.DataFrame(problems, columns=['set', 'product', 'sample', 'error']).to_csv(args.output / 'problems.csv',
                                                                              index=False)
    print(f'\n{len(rows)} tables validated, {len(problems)} problems ---> {args.output.resolve()}')
    return 1 if problems else 0


def cmd_fisher(args):
    import pandas as pd
    from .fisher import FisherForecaster
    loader = _loader(args, _store(args))
    specs = _experiment_specs(args)
    forecaster = FisherForecaster(loader, FisherConfig(parameters=tuple(args.fisher_params),
                                                       n_covariance=args.n_covariance,
                                                       n_derivative=args.n_derivative))
    args.output.mkdir(parents=True, exist_ok=True)
    results, tables, convergence = {}, [], []
    for name, spec in specs.items():
        result = forecaster.compute(spec)
        results[name] = result
        (args.output / name).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output / name / 'fisher.npz', fisher=result.fisher, covariance=result.covariance,
                            mean=result.mean, derivatives=result.derivatives, parameters=np.array(result.parameters),
                            blocks=np.array(result.blocks), block_sizes=np.asarray(result.block_sizes, dtype=np.int64))
        table = result.summary().assign(experiment=name, blocks=spec, n_features=result.mean.size,
                                        condition_number=result.condition_number)
        tables.append(table)
        print(f'\n{name} [{spec}] ({result.mean.size} features)\n{table[["param", "sigma_fisher", "relative_error"]].to_string(index=False)}',
              flush=True)
        if args.convergence:
            convergence.append(forecaster.convergence(spec).assign(experiment=name))
        if args.plots:
            import matplotlib
            matplotlib.use('Agg')
            from .plotting import SBIPlotter
            plotter = SBIPlotter()
            (args.output / name).mkdir(exist_ok=True)
            _save_figure(plotter.correlation(result, title=f'{name}: feature correlations (fiducial)'),
                         args.output / name / 'correlation.png')
            if args.convergence:
                _save_figure(plotter.convergence(convergence[-1], title=f'{name}: Fisher convergence'),
                             args.output / name / 'convergence.png')
    comparison = pd.concat(tables, ignore_index=True)
    comparison.to_csv(args.output / 'fisher_comparison.csv', index=False)
    if convergence:
        pd.concat(convergence, ignore_index=True).to_csv(args.output / 'fisher_convergence.csv', index=False)
    if args.plots:
        from .plotting import SBIPlotter, SERIES
        names = list(results)
        for start in range(0, len(names), len(SERIES)):
            chunk = {n: results[n] for n in names[start:start + len(SERIES)]}
            _save_figure(SBIPlotter().fisher_ellipses(chunk), args.output / f'ellipses_{start // len(SERIES)}.png')
    print(f'\n---> {args.output.resolve()}')
    return 0


def cmd_train(args):
    import pandas as pd
    import torch
    from .evaluation import MomentEvaluator, PosteriorEvaluator
    from .fisher import FisherForecaster
    from .inference import SBITrainer
    from .moments import MomentTrainer
    from .pipeline import SBIPipeline

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    loader = _loader(args, _store(args))
    specs = _experiment_specs(args)
    all_blocks = tuple(dict.fromkeys(b for spec in specs.values() for b in loader.block_names(spec)))
    available = loader.available_ids('latin_hypercube', ','.join(all_blocks))
    cube_ids = available if args.n_cubes is None else np.arange(args.n_cubes)
    n_train = args.n_train or int(round(0.9 * len(cube_ids)))
    training = TrainingConfig(n_train=n_train, split_seed=args.split_seed, train_seed=args.train_seed,
                              hidden_features=args.hidden_features, num_transforms=args.num_transforms,
                              batch_size=args.batch_size, learning_rate=args.learning_rate,
                              validation_fraction=args.validation_fraction, patience=args.patience,
                              max_epochs=args.max_epochs, device=device, ensemble_size=args.ensemble,
                              density_estimator=args.density_estimator, embedding_dim=args.embedding_dim,
                              compression=args.compression, show_train_summary=False)
    moment_training = MomentTrainingConfig(n_train=n_train, split_seed=args.split_seed, train_seed=args.train_seed,
                                           hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
                                           batch_size=args.batch_size, learning_rate=args.learning_rate,
                                           validation_fraction=args.validation_fraction, patience=args.patience,
                                           max_epochs=args.max_epochs, device=device)
    evaluation = EvaluationConfig(n_samples=args.n_samples, sample_seed=args.sample_seed)
    forecaster = None
    if args.fisher or args.compression == 'moped':
        forecaster = FisherForecaster(loader, FisherConfig(parameters=tuple(args.fisher_params)))

    single = len(specs) == 1
    name = next(iter(specs)) if single else None
    output = args.output or Path('results') / ('method_comparison' if args.method == 'both' else
                                               (name if args.method == 'npe' else f'mn_{name}') if single else
                                               ('comparison' if args.method == 'npe' else f'{args.method}_comparison'))
    print(f'experiments: {specs}\nmethods: {args.method} | cubes: {len(cube_ids)} | train: {n_train} | '
          f'device: {device} | output: {output}', flush=True)
    data = loader.load(all_blocks, cube_ids)
    print(f'loaded {data.x.shape[0]} cubes x {data.x.shape[1]} features', flush=True)

    methods = ('npe', 'mn') if args.method == 'both' else (args.method,)
    comparisons, fiducial_tables, all_results = [], [], {}
    for method in methods:
        if method == 'npe':
            pipeline = SBIPipeline(loader, SBITrainer(training), PosteriorEvaluator(evaluation),
                                   fisher=forecaster, n_fiducial=args.fiducial)
        else:
            pipeline = SBIPipeline(loader, MomentTrainer(moment_training), MomentEvaluator(evaluation),
                                   fisher=forecaster, n_fiducial=args.fiducial)
        if method == 'mn' and args.compression == 'moped':
            print('note: --compression applies to NPE only; MN uses the raw features.', file=sys.stderr)

        def destination(label, method=method):
            base = output / method if args.method == 'both' else output
            return base if single and args.method != 'both' and not args.compare else base / label

        def save(label, result, method=method):
            where = destination(label)
            result.save_diagnostics(where, save_model=not args.no_save_model)
            if args.plots:
                _plots(result, where, args, method)
            print(f'\n{method}/{label}\n{result.summary.to_string(index=False)}\n'
                  f'scores: {json.dumps(result.evaluation.scores)}', flush=True)
            if result.fiducial is not None:
                cols = [c for c in ('param', 'bias', 'mean_posterior_std', 'sigma_fisher', 'std_over_fisher',
                                    'coverage_68', 'coverage_95') if c in result.fiducial_summary().columns]
                print(f'fiducial ({len(result.fiducial.truth)} realizations):\n'
                      f'{result.fiducial_summary()[cols].to_string(index=False)}', flush=True)
            print(f'---> results: {where.resolve()}', flush=True)

        results = pipeline.run_many(specs, data=data, callback=save)
        all_results.update({(method, k): v for k, v in results.items()})
        comparisons.append(pipeline.compare(results))
        fiducial = pipeline.compare_fiducial(results)
        if fiducial is not None:
            fiducial_tables.append(fiducial)
    if len(specs) > 1 or args.method == 'both':
        output.mkdir(parents=True, exist_ok=True)
        comparison = pd.concat(comparisons, ignore_index=True)
        comparison.to_csv(output / 'comparison.csv', index=False)
        if fiducial_tables:
            pd.concat(fiducial_tables, ignore_index=True).to_csv(output / 'fiducial_comparison.csv', index=False)
        if args.plots:
            _comparison_plots(all_results, comparison, output)
        print(f'\n---> comparison: {(output / "comparison.csv").resolve()}')
    return 0


def _plots(result, where, args, method):
    import matplotlib
    matplotlib.use('Agg')
    from .plotting import SBIPlotter
    plotter = SBIPlotter()
    prediction, coverage = plotter.diagnostics(result)
    _save_figure(prediction, where / 'prediction.png')
    _save_figure(coverage, where / 'coverage.png')
    _save_figure(plotter.calibration(result.evaluation, f'{result.name}: test calibration'), where / 'calibration.png')
    _save_figure(plotter.sbc(result.evaluation, title=f'{result.name}: SBC ranks (test)'), where / 'sbc.png')
    if method == 'npe':
        _save_figure(plotter.tarp(result.evaluation, title=result.name), where / 'tarp.png')
        _save_figure(plotter.posterior(result, n_samples=args.n_samples, seed=args.sample_seed),
                     where / 'posterior.png')
    if result.fiducial is not None:
        _save_figure(plotter.fiducial_scatter(result), where / 'fiducial_scatter.png')
        _save_figure(plotter.calibration(result.fiducial, f'{result.name}: fiducial calibration'),
                     where / 'fiducial_calibration.png')
        if method == 'npe' and result.fiducial_x is not None:
            _save_figure(plotter.fiducial_posterior(result, result.fiducial_x[0], n_samples=args.n_samples,
                                                    seed=args.sample_seed), where / 'fiducial_posterior.png')
    if result.fisher is not None:
        _save_figure(plotter.correlation(result.fisher, title=f'{result.name}: feature correlations (fiducial)'),
                     where / 'correlation.png')


def _comparison_plots(all_results, comparison, output):
    import matplotlib
    matplotlib.use('Agg')
    from .plotting import SBIPlotter
    plotter = SBIPlotter()
    for metric in ('RMSE', 'coverage_68', 'mean_posterior_std', 'R2'):
        _save_figure(plotter.compare_metrics(comparison, metric), output / f'comparison_{metric}.png')
    for method in dict.fromkeys(m for m, _ in all_results):
        results = {name: r for (m, name), r in all_results.items() if m == method}
        if any(r.fiducial is not None or r.fisher is not None for r in results.values()):
            _save_figure(plotter.compare_errors(results), output / f'errors_{method}.png')
        fishers = {name: r.fisher for name, r in results.items() if r.fisher is not None}
        if fishers and method == 'npe':
            from .plotting import SERIES
            names = list(fishers)
            for start in range(0, len(names), len(SERIES)):
                chunk = {n: fishers[n] for n in names[start:start + len(SERIES)]}
                _save_figure(plotter.fisher_ellipses(chunk), output / f'fisher_ellipses_{start // len(SERIES)}.png')


def cmd_replot(args):
    from .replot import replot
    failed = 0
    for path in args.paths:
        report = replot(path, with_model=not args.no_model, n_samples=args.n_samples, device=args.device)
        for status, what in report:
            if status != 'ok':
                print(f'{status:5s} {what}', flush=True)
        done = sum(s == 'ok' for s, _ in report)
        failed += sum(s == 'error' for s, _ in report)
        print(f'{path}: {done} figures redrawn', flush=True)
    return 1 if failed else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and argv[0] not in ('-h', '--help'):
        argv = ['train'] + argv  # old interface
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return {'inventory': cmd_inventory, 'validate': cmd_validate, 'fisher': cmd_fisher,
                'train': cmd_train, 'replot': cmd_replot}[args.command](args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
