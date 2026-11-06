'''Neural posterior estimation (NPE) with deep ensembles.

Each ensemble member is an independent conditional flow trained with its own
seed, and therefore its own internal validation subset for early stopping.
The ensemble posterior is the equal-weight mixture of the members, each
truncated to the prior box (which is how sbi samples them). Mixtures of
independently trained flows are markedly better calibrated than a single
flow when only ~10^3 simulations are available.
'''
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import PriorConfig, TrainingConfig
from .data import DatasetSplit, SpectrumDataset


def _check_device(device: str):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError(f'Device {device!r} requested, but CUDA is unavailable (use --device cpu).')


def _seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainedPosterior:
    '''Equal-weight mixture of sbi posteriors plus the feature preprocessing.

    Inputs ``x`` are always raw feature vectors (as produced by the loader);
    the optional compressor is applied internally.
    '''

    def __init__(self, members: list, prior: PriorConfig, parameter_names, feature_dim: int,
                 compressor=None, device: str = 'cpu', blocks: tuple[str, ...] = ()):
        if not members:
            raise ValueError('An ensemble needs at least one member.')
        self.members = list(members)
        self.prior = prior
        self.parameter_names = tuple(parameter_names)
        self.feature_dim = int(feature_dim)
        self.compressor = compressor
        self.device = device
        self.blocks = tuple(blocks)
        self._low = torch.tensor(prior.low, dtype=torch.float32)
        self._high = torch.tensor(prior.high, dtype=torch.float32)

    # --------------------------------------------------------------- helpers
    def _prepare(self, x) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        single = x.ndim == 1
        x = x.unsqueeze(0) if single else x
        if x.ndim != 2 or x.shape[1] != self.feature_dim or not torch.isfinite(x).all():
            raise ValueError(f'x must be finite with {self.feature_dim} features per row (got {tuple(x.shape)}).')
        if self.compressor is not None:
            x = self.compressor(x.double())
        return x.to(self.device)

    def to(self, device: str) -> 'TrainedPosterior':
        _check_device(device)
        for member in self.members:
            member.to(device)
        self.device = device
        return self

    # --------------------------------------------------------------- sampling
    def sample_batched(self, x, n_samples: int, seed: int | None = None) -> torch.Tensor:
        '''Samples of shape (n_samples, batch, n_parameters) on the CPU.'''
        if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
            raise ValueError('n_samples must be a positive integer.')
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
                raise ValueError('seed must be a nonnegative integer.')
            _seed_everything(seed)
        x = self._prepare(x)
        k = len(self.members)
        counts = [n_samples // k + (i < n_samples % k) for i in range(k)]
        draws = [member.sample_batched((c,), x=x, show_progress_bars=False).detach().cpu()
                 for member, c in zip(self.members, counts) if c > 0]
        samples = torch.cat(draws, dim=0)
        if k > 1:  # interleave members so that any prefix of the samples is a fair mixture draw
            samples = samples[torch.randperm(n_samples)]
        return samples

    def sample(self, x, n_samples: int = 5000, seed: int | None = None) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim != 1:
            raise ValueError('sample() takes one feature vector; use sample_batched() for several.')
        return self.sample_batched(x, n_samples, seed)[:, 0]

    # --------------------------------------------------------------- densities
    def acceptance(self, x, n_samples: int = 2000) -> torch.Tensor:
        '''Fraction of each member's mass inside the prior box, shape (members, batch).'''
        x = self._prepare(x)
        low, high = self._low.to(self.device), self._high.to(self.device)
        out = []
        with torch.no_grad():
            for member in self.members:
                raw = member.posterior_estimator.sample((n_samples,), condition=x)
                out.append(((raw >= low) & (raw <= high)).all(-1).float().mean(0).cpu())
        return torch.stack(out)

    def log_prob(self, theta, x, leakage_samples: int = 2000, acceptance: torch.Tensor | None = None) -> torch.Tensor:
        '''Normalized log q(theta | x) of the mixture for paired rows of theta and x.

        ``acceptance`` (members, batch) can be passed to reuse an estimate from ``acceptance()``.
        '''
        theta = torch.as_tensor(theta, dtype=torch.float32)
        prepared = self._prepare(x)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        if theta.shape != (prepared.shape[0], len(self.parameter_names)):
            raise ValueError('theta must have one row per x and one column per parameter.')
        theta = theta.to(self.device)
        logs = []
        with torch.no_grad():
            for member in self.members:
                logs.append(member.log_prob_batched(theta.unsqueeze(0), x=prepared, norm_posterior=False)[0].cpu())
        logs = torch.stack(logs)
        if acceptance is None and leakage_samples:
            acceptance = self.acceptance(x, leakage_samples)
        if acceptance is not None:
            logs = logs - torch.log(acceptance.clamp_min(1.0 / max(leakage_samples, 1)))
        return torch.logsumexp(logs, dim=0) - np.log(len(self.members))

    # --------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        device = self.device
        self.to('cpu')
        try:
            torch.save({'format': 'quijotes-sbi/trained-posterior/1', 'model': self}, path)
        finally:
            self.to(device)
        return path

    @staticmethod
    def load(path: str | Path, device: str = 'cpu') -> 'TrainedPosterior':
        '''Load a model saved by ``save``. Only load files you created (pickle).'''
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(payload, dict) or payload.get('format') != 'quijotes-sbi/trained-posterior/1':
            raise ValueError(f'{path} is not a saved TrainedPosterior.')
        return payload['model'].to(device)


@dataclass
class FitResult:
    model: TrainedPosterior
    split: DatasetSplit
    training_config: TrainingConfig
    prior_config: PriorConfig
    member_summaries: list[dict] = field(default_factory=list)

    @property
    def posterior(self) -> Any:
        '''Raw sbi posterior (one member) or sbi EnsemblePosterior; expects *compressed* x if compression is on.'''
        if len(self.model.members) == 1:
            return self.model.members[0]
        from sbi.inference.posteriors import EnsemblePosterior
        return EnsemblePosterior(self.model.members)

    def sample(self, x, n_samples: int = 5000, seed: int | None = None) -> torch.Tensor:
        '''Sample p(theta | x); x is one raw feature vector in the training layout.'''
        return self.model.sample(x, n_samples, seed)


class SBITrainer:
    '''Train an ensemble of NPE flows with the uniform latin-hypercube prior.'''

    def __init__(self, config: TrainingConfig | None = None, prior: PriorConfig | None = None):
        self.config = config or TrainingConfig()
        self.prior = prior or PriorConfig()

    def _builder(self, input_dim: int):
        from sbi.neural_nets import posterior_nn
        from sbi.neural_nets.embedding_nets import FCEmbedding

        cfg = self.config
        kwargs = dict(z_score_x='independent', z_score_theta='independent', hidden_features=cfg.hidden_features)
        if cfg.embedding_dim:
            kwargs['embedding_net'] = FCEmbedding(input_dim=input_dim, output_dim=cfg.embedding_dim,
                                                  num_layers=cfg.embedding_layers, num_hiddens=cfg.embedding_hidden)
        if cfg.density_estimator == 'mdn':
            kwargs['num_components'] = 10
        else:
            kwargs['num_transforms'] = cfg.num_transforms
        return posterior_nn(model=cfg.density_estimator, **kwargs)

    def fit(self, data: SpectrumDataset, compressor=None) -> FitResult:
        from sbi.inference import NPE
        from sbi.utils import BoxUniform

        cfg = self.config
        _check_device(cfg.device)
        if (cfg.compression == 'moped') != (compressor is not None):
            raise ValueError("compression='moped' requires a ScoreCompressor (and only then).")
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

        x_train = split.x_train
        if compressor is not None:
            if compressor.input_dim != x_train.shape[1]:
                raise ValueError('Compressor was built for a different feature layout.')
            x_train = compressor(x_train.double())
        spread = x_train.std(dim=0)
        constant = torch.nonzero(~(spread > 1e-8 * x_train.abs().mean(dim=0).clamp_min(1e-12))).flatten()
        if len(constant):
            raise ValueError(f'Features {constant.tolist()[:10]} are constant over the training cubes; '
                             'drop them (e.g. adjust k range or bins).')

        members, summaries = [], []
        for member in range(cfg.ensemble_size):
            seed = cfg.train_seed + 1000 * member
            _seed_everything(seed)
            prior = BoxUniform(low=low, high=high, device=cfg.device)
            inference = NPE(prior=prior, density_estimator=self._builder(x_train.shape[1]), device=cfg.device,
                            show_progress_bars=False)
            estimator = inference.append_simulations(split.theta_train.to(cfg.device), x_train.to(cfg.device)).train(
                training_batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
                validation_fraction=cfg.validation_fraction, stop_after_epochs=cfg.patience,
                max_num_epochs=cfg.max_epochs, clip_max_norm=cfg.clip_max_norm,
                show_train_summary=cfg.show_train_summary)
            summary = inference._summary
            members.append(inference.build_posterior(estimator))
            summaries.append({'seed': seed, 'epochs_trained': int(summary['epochs_trained'][-1]),
                              'best_validation_loss': float(summary['best_validation_loss'][-1])})
            del inference
        model = TrainedPosterior(members, self.prior, data.parameter_names, data.x.shape[1],
                                 compressor=compressor, device=cfg.device, blocks=data.envs)
        return FitResult(model, split, cfg, self.prior, summaries)
