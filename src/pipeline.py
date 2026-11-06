'''Orchestration: individual blocks, combinations, Fisher reference and fiducial checks.'''
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import json
import platform
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .config import PARAMETER_NAMES
from .data import SpectrumDataset
from .evaluation import EvaluationResult, MomentEvaluator, PosteriorEvaluator
from .fisher import FisherForecaster, FisherResult, ScoreCompressor
from .inference import FitResult, SBITrainer
from .moments import MomentFitResult, MomentTrainer


def provenance() -> dict:
    '''Code version and environment, to make every result traceable.'''
    import sbi
    import torch
    root = Path(__file__).resolve().parents[1]
    info = {'time_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'host': platform.node(),
            'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
            'sbi': sbi.__version__}
    try:
        info['git_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                                            text=True, timeout=10).stdout.strip()
        info['git_dirty'] = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root,
                                                capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class ExperimentResult:
    fit: FitResult | MomentFitResult
    evaluation: EvaluationResult
    name: str = ''
    fiducial: EvaluationResult | None = None
    fisher: FisherResult | None = None
    extra: dict = field(default_factory=dict)
    fiducial_x: object = None  # features of the fiducial realizations evaluated (not saved)

    @property
    def method(self) -> str:
        return 'mn' if isinstance(self.fit, MomentFitResult) else 'npe'

    @property
    def summary(self):
        return self.evaluation.summary

    @property
    def posterior(self):
        if isinstance(self.fit, MomentFitResult):
            raise ValueError('Moment Networks do not estimate a posterior density; use fit.predict_moments(x).')
        return self.fit.posterior

    def fiducial_summary(self):
        '''Fiducial-realization metrics, next to the Fisher (Cramer-Rao-like) errors when available.'''
        if self.fiducial is None:
            return None
        table = self.fiducial.summary
        mean = self.fiducial.means.mean(axis=0)
        truth = self.fiducial.truth[0]
        table.insert(1, 'fiducial', truth)
        table.insert(2, 'mean_of_posterior_means', mean)
        table.insert(3, 'scatter_of_posterior_means', self.fiducial.means.std(axis=0, ddof=1))
        if self.fisher is not None:
            sigma = dict(zip(self.fisher.parameters, self.fisher.sigma))
            table['sigma_fisher'] = [sigma.get(p, np.nan) for p in table.param]
            table['std_over_fisher'] = table.mean_posterior_std / table.sigma_fisher
        return table

    def save_diagnostics(self, directory: str | Path, save_model: bool = True) -> Path:
        '''Save metrics, calibration curves, arrays, configuration, provenance and the model.'''
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        ev = self.evaluation
        self.summary.to_csv(directory / 'summary.csv', index=False)
        if ev.pit is not None:
            ev.coverage_curve().to_csv(directory / 'calibration.csv', index=False)
        scores = {'test': ev.scores}
        diagnostics = ev.arrays()
        diagnostics['train_cube_ids'] = self.fit.split.train_cube_ids
        diagnostics['test_cube_ids'] = ev.cube_ids
        if self.fiducial is not None:
            self.fiducial_summary().to_csv(directory / 'fiducial_summary.csv', index=False)
            scores['fiducial'] = self.fiducial.scores
            diagnostics.update({f'fiducial_{k}': v for k, v in self.fiducial.arrays().items()})
        if self.fisher is not None:
            self.fisher.summary().to_csv(directory / 'fisher.csv', index=False)
            diagnostics['fisher_matrix'] = self.fisher.fisher
        if isinstance(self.fit, MomentFitResult):
            diagnostics['oof_means'] = self.fit.oof_means.numpy()
            diagnostics['oof_fold'] = self.fit.oof_fold.numpy()
            diagnostics['oof_squared_residuals'] = self.fit.oof_squared_residuals.numpy()
        np.savez_compressed(directory / 'diagnostics.npz', **diagnostics)
        (directory / 'scores.json').write_text(json.dumps(_jsonable(scores), indent=2), encoding='utf-8')

        data = self.fit.split.data
        metadata = {'name': self.name, 'method': self.method, 'interval_method': ev.interval_method,
                    'blocks': data.envs, 'envs': data.envs, 'parameter_names': data.parameter_names,
                    'n_features': int(data.x.shape[1]), 'n_train': int(len(self.fit.split.train_rows)),
                    'n_test': int(len(self.fit.split.test_rows)),
                    'features': _jsonable(data.config), 'spectrum': _jsonable(getattr(data.config, 'spectrum',
                                                                                    data.config)),
                    'training': _jsonable(self.fit.training_config),
                    'prior': _jsonable(self.fit.prior_config) if isinstance(self.fit, FitResult) else None,
                    'evaluation': _jsonable(ev.config),
                    'k': {env: k.tolist() for env, k in data.k.items()},
                    'feature_slices': {env: [s.start, s.stop] for env, s in data.feature_slices.items()},
                    'provenance': provenance(), **_jsonable(self.extra)}
        if isinstance(self.fit, FitResult):
            metadata['ensemble'] = self.fit.member_summaries
            metadata['compression'] = self.fit.training_config.compression
            if save_model:
                self.fit.model.save(directory / 'model.pt')
                metadata['model_file'] = 'model.pt'
        else:
            metadata['variance_targets'] = 'five_fold_out_of_fold_squared_residuals'
            metadata['oof_folds'] = 5
            metadata['oof_best_epochs'] = self.fit.oof_best_epochs
            metadata['mean_best_epoch'] = self.fit.mean_best_epoch
            metadata['variance_best_epoch'] = self.fit.variance_best_epoch
        if self.fiducial is not None:
            metadata['fiducial_ids'] = self.fiducial.cube_ids.tolist()
        (directory / 'metadata.json').write_text(json.dumps(_jsonable(metadata), indent=2), encoding='utf-8')
        return directory


class SBIPipeline:
    '''Train/evaluate one model per experiment, with the same cube split for all.

    ``fisher``: a FisherForecaster to compute the Gaussian reference (and the
    MOPED compressor when the trainer asks for it). ``n_fiducial``: number of
    fiducial realizations on which to validate the posterior at a fixed,
    known cosmology. Both need a loader with ``load_set`` (QuijoteSummaryLoader).
    '''

    def __init__(self, loader, trainer: SBITrainer | MomentTrainer | None = None,
                 evaluator: PosteriorEvaluator | MomentEvaluator | None = None,
                 fisher: FisherForecaster | None = None, n_fiducial: int = 0):
        self.loader = loader
        self.trainer = trainer or SBITrainer()
        self.evaluator = evaluator or (MomentEvaluator() if isinstance(self.trainer, MomentTrainer)
                                       else PosteriorEvaluator())
        self.fisher = fisher
        self.n_fiducial = int(n_fiducial)
        if (fisher is not None or self.n_fiducial) and not hasattr(loader, 'load_set'):
            raise ValueError('Fisher forecasts and fiducial checks need a QuijoteSummaryLoader.')
        if self._wants_compression and fisher is None:
            raise ValueError("compression='moped' needs a FisherForecaster.")
        self._fisher_cache: dict = {}

    @property
    def _wants_compression(self) -> bool:
        return isinstance(self.trainer, SBITrainer) and self.trainer.config.compression == 'moped'

    def _fisher(self, blocks, parameters):
        key = (tuple(blocks), tuple(parameters))
        if key not in self._fisher_cache:
            self._fisher_cache[key] = self.fisher.compute(','.join(blocks), parameters=parameters)
        return self._fisher_cache[key]

    def _run_one(self, name: str, data: SpectrumDataset) -> ExperimentResult:
        fisher = compressor = None
        extra = {}
        if self.fisher is not None:
            fisher = self._fisher(data.envs, self.fisher.config.parameters)
        if self._wants_compression:
            compression_fisher = self._fisher(data.envs, data.parameter_names)
            compressor = ScoreCompressor.from_fisher(compression_fisher)
            extra['compressor'] = {'type': 'moped_score', 'parameters': compressor.parameters,
                                   'n_covariance': compression_fisher.n_covariance,
                                   'n_derivative': compression_fisher.n_derivative}
            fit = self.trainer.fit(data, compressor=compressor)
        else:
            fit = self.trainer.fit(data)
        evaluation = self.evaluator.evaluate(fit)
        fiducial = fiducial_x = None
        if self.n_fiducial:
            fixed = self.loader.load_set('fiducial', data.envs)
            n = min(self.n_fiducial, len(fixed.ids))
            fiducial_x = fixed.x_tensor[:n]
            fiducial = self.evaluator.evaluate_fixed(fit, fiducial_x, fixed.theta(data.parameter_names),
                                                     fixed.ids[:n])
        return ExperimentResult(fit, evaluation, name=name, fiducial=fiducial, fisher=fisher, extra=extra,
                                fiducial_x=fiducial_x)

    def run(self, envs: str | Iterable[str], cube_ids: Iterable[int] | None = None) -> ExperimentResult:
        data = self.loader.load(envs, cube_ids)
        return self._run_one('_'.join(data.envs), data)

    def run_many(self, experiments: Mapping[str, str | Iterable[str]], cube_ids: Iterable[int] | None = None,
                 data: SpectrumDataset | None = None, callback=None) -> dict[str, ExperimentResult]:
        '''Load every block once and use the same cube split for every fit.

        ``callback(name, result)`` is called after each experiment (e.g. to save
        results incrementally so a crash does not lose finished experiments).
        '''
        if not experiments:
            raise ValueError('Provide at least one experiment.')
        normalize = getattr(self.loader, 'block_names', _names)
        groups = {name: normalize(blocks) for name, blocks in experiments.items()}
        if any(not group or len(set(group)) != len(group) for group in groups.values()):
            raise ValueError('Each experiment requires nonempty, distinct blocks.')
        all_blocks = tuple(dict.fromkeys(b for group in groups.values() for b in group))
        if data is None:
            data = self.loader.load(all_blocks, cube_ids)
        elif not set(all_blocks) <= set(data.envs) or (
                cube_ids is not None and sorted(data.cube_ids.tolist()) != sorted(cube_ids)):
            raise ValueError('Preloaded data must contain the requested blocks and cube IDs.')
        results = {}
        for name, blocks in groups.items():
            results[name] = self._run_one(name, data.select(blocks))
            if callback is not None:
                callback(name, results[name])
        return results

    @staticmethod
    def compare(results: Mapping[str, ExperimentResult]):
        '''Test-set metrics of every experiment in one table.'''
        import pandas as pd
        if not results:
            raise ValueError('No results to compare.')
        return pd.concat([r.summary.assign(experiment=name, method=r.method,
                                           interval_method=r.evaluation.interval_method,
                                           nll=r.evaluation.scores.get('nll', np.nan),
                                           tarp_max_deviation=r.evaluation.scores.get('tarp_max_deviation', np.nan))
                          for name, r in results.items()], ignore_index=True)

    @staticmethod
    def compare_fiducial(results: Mapping[str, ExperimentResult]):
        import pandas as pd
        tables = [r.fiducial_summary().assign(experiment=name, method=r.method)
                  for name, r in results.items() if r.fiducial is not None]
        return pd.concat(tables, ignore_index=True) if tables else None


def _names(blocks) -> tuple[str, ...]:
    if isinstance(blocks, str):
        return tuple(b.strip() for b in blocks.split(',') if b.strip())
    out = []
    for b in blocks:
        out.append(b.name if hasattr(b, 'name') else str(b))
    return tuple(out)
