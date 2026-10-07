from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch

from .config import PriorConfig, TrainingConfig
from .data import DatasetSplit, SpectrumDataset


@dataclass
class FitResult:
    posterior: Any
    inference: Any
    density_estimator: Any
    split: DatasetSplit
    training_config: TrainingConfig
    prior_config: PriorConfig

    def sample(self, x, n_samples: int = 5000, seed: int | None = None) -> torch.Tensor:
        '''Sample p(theta | x); x is a single already-transformed feature vector.'''
        if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
            raise ValueError('n_samples must be a positive integer.')
        x = torch.as_tensor(x, dtype=torch.float32, device=self.training_config.device)

        if x.shape != (self.split.data.x.shape[1],) or not torch.isfinite(x).all():
            raise ValueError('x must be a finite vector with the training feature dimension.')

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
                raise ValueError('seed must be a nonnegative integer.')
            torch.manual_seed(seed)
        return self.posterior.sample((n_samples,), x=x, show_progress_bars=False).detach().cpu()


class SBITrainer:
    '''Train the NSF posterior with the uniform Quijote prior.'''
    def __init__(self, config: TrainingConfig | None = None, prior: PriorConfig | None = None):
        self.config = config or TrainingConfig()
        self.prior = prior or PriorConfig()

    def fit(self, data: SpectrumDataset) -> FitResult:
        from sbi.inference import NPE
        from sbi.neural_nets import posterior_nn
        from sbi.utils import BoxUniform

        cfg = self.config
        split = data.split(cfg.n_train, cfg.split_seed)
        n_validation = int(cfg.n_train * cfg.validation_fraction)

        if n_validation < 1 or cfg.n_train - n_validation < 2:
            raise ValueError('Split must provide at least 1 validation and 2 training cubes.')

        if data.theta.shape[1] != len(self.prior.low):
            raise ValueError('Prior dimension must match theta columns.')

        low = torch.tensor(self.prior.low, dtype=torch.float32)
        high = torch.tensor(self.prior.high, dtype=torch.float32)
        theta_cpu = data.theta.detach().cpu()

        if torch.any(theta_cpu < low) or torch.any(theta_cpu > high):
            raise ValueError('Cosmological parameters lie outside the configured prior.')

        torch.manual_seed(cfg.train_seed)
        prior = BoxUniform(low=low, high=high, device=cfg.device)
        builder = posterior_nn(model='nsf', hidden_features=cfg.hidden_features, num_transforms=cfg.num_transforms,
                               z_score_x='independent', z_score_theta='independent')
        inference = NPE(prior=prior, density_estimator=builder, device=cfg.device)
        density_estimator = inference.append_simulations(split.theta_train.to(cfg.device),
                                                         split.x_train.to(cfg.device)).train(training_batch_size=cfg.batch_size,
                                                                                             learning_rate=cfg.learning_rate,
                                                                                             validation_fraction=cfg.validation_fraction,
                                                                                             stop_after_epochs=cfg.patience,
                                                                                             max_num_epochs=cfg.max_epochs,
                                                                                             show_train_summary=cfg.show_train_summary)
        return FitResult(posterior=inference.build_posterior(density_estimator), inference=inference,
                         density_estimator=density_estimator, split=split, training_config=cfg, prior_config=self.prior)