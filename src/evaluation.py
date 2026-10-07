from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import EvaluationConfig
from .inference import FitResult

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class EvaluationResult:
    truth: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    low68: np.ndarray
    high68: np.ndarray
    low95: np.ndarray
    high95: np.ndarray
    cube_ids: np.ndarray
    parameter_names: tuple[str, ...]
    config: EvaluationConfig

    @property
    def summary(self) -> pd.DataFrame:
        import pandas as pd

        errors = self.means - self.truth
        return pd.DataFrame({'param': self.parameter_names,
                             'bias': errors.mean(axis=0),
                             'RMSE': np.sqrt((errors ** 2).mean(axis=0)),
                             'mean_posterior_std': self.stds.mean(axis=0),
                             'mean_68_interval_width': (self.high68 - self.low68).mean(axis=0),
                             'coverage_68': ((self.truth >= self.low68) & (self.truth <= self.high68)).mean(axis=0),
                             'coverage_95': ((self.truth >= self.low95) & (self.truth <= self.high95)).mean(axis=0)})


class PosteriorEvaluator:
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def evaluate(self, fit: FitResult) -> EvaluationResult:
        truth = fit.split.theta_test.detach().cpu().numpy()
        x_test = fit.split.x_test
        test_cube_ids = fit.split.test_cube_ids
        means = np.empty(truth.shape, dtype=float)
        stds = np.empty_like(means)
        intervals = np.empty((4, *truth.shape), dtype=float)
        torch.manual_seed(self.config.sample_seed)

        for j in tqdm(range(len(truth)), desc='test', disable=not self.config.show_progress):
            samples = fit.sample(x_test[j], self.config.n_samples).numpy()
            if samples.shape != (self.config.n_samples, truth.shape[1]) or not np.isfinite(samples).all():
                raise ValueError(f'Invalid posterior samples for cube {test_cube_ids[j]}.')
            means[j] = samples.mean(axis=0)
            stds[j] = samples.std(axis=0, ddof=1)
            intervals[:, j, :] = np.quantile(samples, [0.025, 0.16, 0.84, 0.975], axis=0)
        low95, low68, high68, high95 = intervals

        return EvaluationResult(truth=truth, means=means, stds=stds, low68=low68, high68=high68,
                                low95=low95, high95=high95, cube_ids=test_cube_ids.copy(),
                                parameter_names=fit.split.data.parameter_names, config=self.config)