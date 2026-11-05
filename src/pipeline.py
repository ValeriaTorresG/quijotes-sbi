'''Orchestration for individual environments and concatenated spectra.'''
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .data import QuijoteDataLoader, SpectrumDataset
from .evaluation import EvaluationResult, MomentEvaluator, PosteriorEvaluator
from .inference import FitResult, SBITrainer
from .moments import MomentFitResult, MomentTrainer


@dataclass
class ExperimentResult:
    fit: FitResult | MomentFitResult
    evaluation: EvaluationResult

    @property
    def summary(self):
        return self.evaluation.summary

    @property
    def posterior(self):
        if isinstance(self.fit, MomentFitResult):
            raise ValueError('Moment Networks do not estimate a posterior density; use fit.predict_moments(x).')
        return self.fit.posterior

    def save_diagnostics(self, directory: str | Path) -> Path:
        '''Save metrics, cube IDs, features metadata, and reproducibility configs.'''
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(directory / 'summary.csv', index=False)

        ev = self.evaluation
        diagnostics = dict(truth=ev.truth, means=ev.means, stds=ev.stds,
                           low68=ev.low68, high68=ev.high68, low95=ev.low95, high95=ev.high95,
                           train_cube_ids=self.fit.split.train_cube_ids, test_cube_ids=ev.cube_ids)
        if isinstance(self.fit, MomentFitResult):
            diagnostics['oof_means'] = self.fit.oof_means.numpy()
            diagnostics['oof_fold'] = self.fit.oof_fold.numpy()
            diagnostics['oof_squared_residuals'] = self.fit.oof_squared_residuals.numpy()
        np.savez_compressed(directory / 'diagnostics.npz', **diagnostics)

        data = self.fit.split.data
        metadata = {'method': 'mn' if isinstance(self.fit, MomentFitResult) else 'npe',
                    'interval_method': ev.interval_method, 'envs': data.envs, 'parameter_names': data.parameter_names,
                    'spectrum': asdict(data.config), 'training': asdict(self.fit.training_config),
                    'prior': asdict(self.fit.prior_config) if isinstance(self.fit, FitResult) else None,
                    'evaluation': asdict(ev.config),
                    'k': {env: k.tolist() for env, k in data.k.items()},
                    'feature_slices': {env: [s.start, s.stop] for env, s in data.feature_slices.items()}}
        if isinstance(self.fit, MomentFitResult):
            metadata['variance_targets'] = 'five_fold_out_of_fold_squared_residuals'
            metadata['oof_folds'] = 5
            metadata['oof_best_epochs'] = self.fit.oof_best_epochs
            metadata['mean_best_epoch'] = self.fit.mean_best_epoch
            metadata['variance_best_epoch'] = self.fit.variance_best_epoch
        (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return directory


class SBIPipeline:
    '''Components can also be used independently, without this orchestrator.'''

    def __init__(self, loader: QuijoteDataLoader,
                 trainer: SBITrainer | MomentTrainer | None = None,
                 evaluator: PosteriorEvaluator | MomentEvaluator | None = None):
        self.loader = loader
        self.trainer = trainer or SBITrainer()
        self.evaluator = evaluator or (MomentEvaluator() if isinstance(self.trainer, MomentTrainer)
                                       else PosteriorEvaluator())

    def run(self, envs: str | Iterable[str], cube_ids: Iterable[int]) -> ExperimentResult:
        fit = self.trainer.fit(self.loader.load(envs, cube_ids))
        return ExperimentResult(fit, self.evaluator.evaluate(fit))

    def run_many(self, experiments: Mapping[str, str | Iterable[str]], cube_ids: Iterable[int],
                 data: SpectrumDataset | None = None) -> dict[str, ExperimentResult]:
        '''Load each environment once and use the same cube split for every fit.'''
        if not experiments:
            raise ValueError('Provide at least one experiment.')

        groups = {name: (envs,) if isinstance(envs, str) else tuple(envs)
                  for name, envs in experiments.items()}

        if any(not group or len(set(group)) != len(group) for group in groups.values()):
            raise ValueError('Each experiment requires nonempty, distinct environments.')

        all_envs = tuple(dict.fromkeys(env for group in groups.values() for env in group))
        if data is None:
            data = self.loader.load(all_envs, cube_ids)
        elif (tuple(data.envs) != all_envs or
              sorted(data.cube_ids.tolist()) != sorted(cube_ids)):
            raise ValueError('Preloaded data must match the requested environments and cube IDs.')
        results = {}
        for name, envs in groups.items():
            fit = self.trainer.fit(data.select(envs))
            results[name] = ExperimentResult(fit, self.evaluator.evaluate(fit))
        return results

    @staticmethod
    def compare(results: Mapping[str, ExperimentResult]):
        '''Return one table with notebook metrics and an experiment column.'''
        import pandas as pd

        if not results:
            raise ValueError('No results to compare.')
        return pd.concat([result.summary.assign(experiment=name, interval_method=result.evaluation.interval_method)
                          for name, result in results.items()],
                         ignore_index=True)