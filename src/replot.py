'''Redraw figures from saved results, without retraining or re-reading raw files.

Arrays come from ``diagnostics.npz`` / ``*.csv`` and settings from
``metadata.json``. Two kinds of figures need more than saved arrays and are
drawn only when possible:

* posterior corners (test cube, fiducial realization): sample the saved
  ``model.pt`` on features rebuilt from the validated cache with the run's
  own feature configuration;
* feature correlations: the fiducial covariance, saved by newer runs, or
  recomputed from the cached fiducial features.

Use ``with_model=False`` to skip both.
'''
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .config import FIDUCIAL, EvaluationConfig, FeatureConfig, PriorConfig, SpectrumConfig
from .evaluation import EvaluationResult


def _save(fig, path: Path, report: list):
    import matplotlib.pyplot as plt
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    report.append(('ok', str(path)))


def _known(cls, values: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: tuple(v) if isinstance(v, list) else v for k, v in (values or {}).items() if k in names}


def feature_config(metadata: dict) -> FeatureConfig | None:
    raw = metadata.get('features')
    if not isinstance(raw, dict) or 'spectrum' not in raw:
        return None
    return FeatureConfig(spectrum=SpectrumConfig(**_known(SpectrumConfig, raw['spectrum'])),
                         **{k: v for k, v in _known(FeatureConfig, raw).items() if k != 'spectrum'})


def _evaluation(arrays, metadata, prefix='') -> EvaluationResult | None:
    def get(key):
        return arrays[prefix + key] if prefix + key in arrays else None
    if get('truth') is None:
        return None
    names = tuple(metadata.get('parameter_names') or ())
    prior = metadata.get('prior')
    prior = PriorConfig(**_known(PriorConfig, prior)) if prior else (PriorConfig() if len(names) == 5 else None)
    return EvaluationResult(
        truth=get('truth'), means=get('means'), stds=get('stds'), low68=get('low68'), high68=get('high68'),
        low95=get('low95'), high95=get('high95'),
        cube_ids=get('cube_ids') if get('cube_ids') is not None else get('test_cube_ids'),
        parameter_names=names, config=EvaluationConfig(**_known(EvaluationConfig, metadata.get('evaluation'))),
        interval_method=metadata.get('interval_method', 'posterior_quantiles'), pit=get('pit'),
        log_prob=get('log_prob'), tarp_f=get('tarp_f'), acceptance=get('acceptance'),
        prior_low=None if prior is None else np.asarray(prior.low, float),
        prior_high=None if prior is None else np.asarray(prior.high, float))


def load_experiment(directory: str | Path) -> SimpleNamespace:
    '''Saved experiment: name, method, evaluation, fiducial, fisher, metadata.'''
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    with np.load(directory / 'diagnostics.npz', allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    fisher = None
    if 'fisher_matrix' in arrays and (directory / 'fisher.csv').is_file():
        import pandas as pd
        table = pd.read_csv(directory / 'fisher.csv')
        fisher = SimpleNamespace(parameters=tuple(table.param), fisher=arrays['fisher_matrix'],
                                 sigma=table.sigma_fisher.to_numpy(), fiducial=dict(FIDUCIAL),
                                 covariance=arrays.get('fisher_covariance'),
                                 block_sizes=tuple(arrays['fisher_block_sizes']) if 'fisher_block_sizes' in arrays
                                 else (), blocks=tuple(metadata.get('blocks') or metadata.get('envs') or ()))
    return SimpleNamespace(directory=directory, name=metadata.get('name') or directory.name,
                           method=metadata.get('method', 'npe'), metadata=metadata, arrays=arrays,
                           evaluation=_evaluation(arrays, metadata), fiducial=_evaluation(arrays, metadata, 'fiducial_'),
                           fisher=fisher)


class _Features:
    '''Rebuild model inputs from the cache with a run's feature configuration (cached per config).'''

    def __init__(self):
        self._loaders = {}

    def loader(self, config, data: dict | None = None):
        from .config import DataLayout
        from .features import QuijoteSummaryLoader
        from .store import ProductStore
        data = data or {}
        key = (repr(config), json.dumps(data, sort_keys=True))
        if key not in self._loaders:
            layout = DataLayout(**{k: data[k] for k in ('root', 'params_file') if k in data})
            store = ProductStore(layout, data.get('cache_dir'), show_progress=False)
            # Only the realizations being drawn are needed; completeness was checked when training.
            self._loaders[key] = QuijoteSummaryLoader(store, config, allow_missing=True)
        return self._loaders[key]


def replot_experiment(experiment, plotter, report, features: _Features, with_model=True, n_samples=20_000,
                      seed=123, device='cpu'):
    d, name, ev = experiment.directory, experiment.name, experiment.evaluation
    if ev is None:
        report.append(('skip', f'{d}: no test arrays'))
        return
    _save(plotter.prediction(ev, name), d / 'prediction.png', report)
    _save(plotter.coverage(ev, name), d / 'coverage.png', report)
    if ev.pit is not None:
        _save(plotter.calibration(ev, f'{name}: test calibration'), d / 'calibration.png', report)
        _save(plotter.sbc(ev, title=f'{name}: SBC ranks (test)'), d / 'sbc.png', report)
    if ev.tarp_f is not None:
        _save(plotter.tarp(ev, title=name), d / 'tarp.png', report)
    fid = experiment.fiducial
    if fid is not None:
        _save(plotter.fiducial_scatter(fid, f'{name}: fiducial realizations'), d / 'fiducial_scatter.png', report)
        if fid.pit is not None:
            _save(plotter.calibration(fid, f'{name}: fiducial calibration'), d / 'fiducial_calibration.png', report)
    if not with_model:
        return

    config = feature_config(experiment.metadata)
    blocks = tuple(experiment.metadata.get('blocks') or ())
    if config is None or not blocks:
        report.append(('skip', f'{d}: corners/correlation need a run saved by the current pipeline'))
        return
    loader = features.loader(config, experiment.metadata.get('data'))
    fisher = experiment.fisher
    if fisher is not None:
        if fisher.covariance is None:
            fixed = loader.load_set('fiducial', ','.join(blocks))
            fisher.covariance = np.cov(fixed.x, rowvar=False)
            fisher.block_sizes = fixed.block_sizes
        _save(plotter.correlation(fisher, title=f'{name}: feature correlations (fiducial)'), d / 'correlation.png',
              report)
    model_path = d / 'model.pt'
    if experiment.method != 'npe' or not model_path.is_file():
        return
    from .inference import TrainedPosterior
    from .plotting import marginal_covariance
    model = TrainedPosterior.load(model_path, device=device)
    names = ev.parameter_names
    test_id = int(experiment.arrays['test_cube_ids'][0])
    x = loader.load(','.join(blocks), cube_ids=[test_id]).x[0]
    if x.shape[0] != model.feature_dim:
        report.append(('skip', f'{d}: rebuilt features ({x.shape[0]}) do not match the model ({model.feature_dim})'))
        return
    samples = model.sample(x, n_samples, seed).numpy()
    _save(plotter.corner(samples, names, truth=ev.truth[0], title=f'{name}: test cube {test_id}'),
          d / 'posterior.png', report)
    if fid is not None:
        fixed = loader.load_set('fiducial', ','.join(blocks), ids=[int(fid.cube_ids[0])])
        samples = model.sample(fixed.x_tensor[0], n_samples, seed).numpy()
        overlay = None
        if fisher is not None and set(names) <= set(fisher.parameters):
            overlay = (np.array([fisher.fiducial[n] for n in names]), marginal_covariance(fisher, names))
        _save(plotter.corner(samples, names, truth=fid.truth[0], fisher=overlay,
                             title=f'{name}: fiducial realization'), d / 'fiducial_posterior.png', report)


def replot_comparison(directory: Path, experiments: list, plotter, report):
    import pandas as pd
    from .plotting import COLORS
    comparison = pd.read_csv(directory / 'comparison.csv')
    for metric in ('RMSE', 'coverage_68', 'mean_posterior_std', 'R2'):
        if metric in comparison:
            _save(plotter.compare_metrics(comparison, metric), directory / f'comparison_{metric}.png', report)
    for method in dict.fromkeys(e.method for e in experiments):
        group = [e for e in experiments if e.method == method]
        rows = []
        for e in group:
            sigma = dict(zip(e.fisher.parameters, e.fisher.sigma)) if e.fisher is not None else {}
            for j, p in enumerate(e.evaluation.parameter_names):
                rows.append(dict(experiment=e.name, param=p, fisher_sigma=sigma.get(p, np.nan),
                                 npe_std=e.fiducial.stds[:, j].mean() if e.fiducial is not None else np.nan))
        table = pd.DataFrame(rows)
        if len(table) and table[['npe_std', 'fisher_sigma']].notna().any().any():
            _save(plotter.compare_errors(table, method_label=method.upper()), directory / f'errors_{method}.png',
                  report)
        fishers = {e.name: e.fisher for e in group if e.fisher is not None}
        if fishers and method == 'npe':
            names = list(fishers)
            for start in range(0, len(names), len(COLORS)):
                chunk = {n: fishers[n] for n in names[start:start + len(COLORS)]}
                _save(plotter.fisher_ellipses(chunk), directory / f'fisher_ellipses_{start // len(COLORS)}.png',
                      report)


def replot_fisher(directory: Path, plotter, report):
    import pandas as pd
    from .plotting import COLORS
    convergence = directory / 'fisher_convergence.csv'
    if convergence.is_file():
        table = pd.read_csv(convergence)
        for name, rows in table.groupby('experiment', sort=False):
            (directory / name).mkdir(exist_ok=True)
            _save(plotter.convergence(rows, title=f'{name}: Fisher convergence'), directory / name / 'convergence.png',
                  report)
    fishers = {}
    for path in sorted(directory.glob('*/fisher.npz')):
        with np.load(path, allow_pickle=False) as data:
            f = SimpleNamespace(parameters=tuple(data['parameters']), fisher=data['fisher'],
                                covariance=data['covariance'], blocks=tuple(data['blocks']),
                                block_sizes=tuple(data['block_sizes']), fiducial=dict(FIDUCIAL))
        fishers[path.parent.name] = f
        _save(plotter.correlation(f, title=f'{path.parent.name}: feature correlations (fiducial)'),
              path.parent / 'correlation.png', report)
    if fishers:
        names = list(fishers)
        for start in range(0, len(names), len(COLORS)):
            chunk = {n: fishers[n] for n in names[start:start + len(COLORS)]}
            _save(plotter.fisher_ellipses(chunk), directory / f'ellipses_{start // len(COLORS)}.png', report)
    elif (directory / 'fisher_comparison.csv').is_file():
        report.append(('skip', f'{directory}: no fisher.npz saved (older run); ellipses/correlations need '
                               f'`python -m src fisher ... --plots` again'))


def replot(root: str | Path, with_model: bool = True, n_samples: int = 20_000, seed: int = 123,
           device: str = 'cpu', plotter=None) -> list[tuple[str, str]]:
    '''Redraw every figure under ``root``. Returns (status, path or reason) rows.'''
    import matplotlib
    matplotlib.use('Agg')
    from .plotting import SBIPlotter
    plotter = plotter or SBIPlotter()
    root = Path(root)
    report: list = []
    features = _Features()
    experiments = {}
    for meta in sorted(root.rglob('metadata.json')):
        d = meta.parent
        if not (d / 'diagnostics.npz').is_file():
            report.append(('skip', f'{d}: no diagnostics.npz (still running?)'))
            continue
        try:
            experiment = load_experiment(d)
            experiments[d] = experiment
            replot_experiment(experiment, plotter, report, features, with_model, n_samples, seed, device)
        except Exception as error:  # keep going: one broken folder must not stop the rest
            report.append(('error', f'{d}: {type(error).__name__}: {error}'))
    for comparison in sorted(root.rglob('comparison.csv')):
        d = comparison.parent
        children = [e for p, e in experiments.items() if d in p.parents and e.evaluation is not None]
        try:
            replot_comparison(d, children, plotter, report)
        except Exception as error:
            report.append(('error', f'{d}: {type(error).__name__}: {error}'))
    for fisher_dir in {p.parent for p in root.rglob('fisher_comparison.csv')}:
        try:
            replot_fisher(fisher_dir, plotter, report)
        except Exception as error:
            report.append(('error', f'{fisher_dir}: {type(error).__name__}: {error}'))
    return report


def sample_fiducial(experiment, realization: int, n_samples: int = 50_000, seed: int = 123, device: str = 'cpu',
                    features: _Features | None = None) -> np.ndarray:
    '''Samples of a saved NPE model for one fiducial realization (features rebuilt from the cache).'''
    from .inference import TrainedPosterior
    if experiment.method != 'npe' or not (experiment.directory / 'model.pt').is_file():
        raise ValueError(f'{experiment.directory}: no saved NPE model.')
    config = feature_config(experiment.metadata)
    if config is None:
        raise ValueError(f'{experiment.directory}: run saved by an older pipeline (no feature configuration).')
    features = features or _Features()
    loader = features.loader(config, experiment.metadata.get('data'))
    model = TrainedPosterior.load(experiment.directory / 'model.pt', device=device)
    x = loader.load_set('fiducial', ','.join(experiment.metadata['blocks']), ids=[realization]).x_tensor[0]
    return model.sample(x, n_samples, seed).numpy()
