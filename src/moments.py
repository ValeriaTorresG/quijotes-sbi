"""Two-stage Moment Networks with out-of-fold targets for the variance stage.

The mean estimates E[theta|x]. The variance network regresses squared residuals
from five held-out predictions; it does not estimate a joint density or quantiles.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import MomentTrainingConfig
from .data import DatasetSplit, SpectrumDataset


def _mlp(n_input: int, n_output: int, width: int, depth: int, positive: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(depth):
        layers += [nn.Linear(n_input if i == 0 else width, width), nn.ReLU()]
    layers.append(nn.Linear(width, n_output))
    if positive:
        layers.append(nn.Softplus())
    return nn.Sequential(*layers)


def _scaling(x: torch.Tensor, theta: torch.Tensor, rows: torch.Tensor):
    x_mean = x[rows].mean(0)
    x_scale = x[rows].std(0, unbiased=False).clamp_min(1e-6)
    theta_mean = theta[rows].mean(0)
    theta_scale = theta[rows].std(0, unbiased=False).clamp_min(1e-6)
    return x_mean, x_scale, theta_mean, theta_scale


def _normalized(x: torch.Tensor, target: torch.Tensor, scaling, kind: str):
    x_mean, x_scale, theta_mean, theta_scale = scaling
    normalized_x = (x - x_mean) / x_scale
    normalized_target = ((target - theta_mean) / theta_scale if kind == 'mean'
                         else target / theta_scale.square())
    return normalized_x, normalized_target


def _loader(x: torch.Tensor, target: torch.Tensor, rows: torch.Tensor,
            config: MomentTrainingConfig, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x[rows].cpu(), target[rows].cpu()),
                      batch_size=config.batch_size, shuffle=True, generator=generator)


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer, device: str):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(xb.to(device)) - yb.to(device)) ** 2)
        if not torch.isfinite(loss):
            raise ValueError('Moment Network training loss became non-finite.')
        loss.backward()
        optimizer.step()


def _new_model(x: torch.Tensor, target: torch.Tensor, config: MomentTrainingConfig,
               kind: str, seed: int):
    torch.manual_seed(seed)
    if config.device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    return _mlp(x.shape[1], target.shape[1], config.hidden_features,
                config.hidden_layers, positive=(kind == 'variance')).to(config.device)


def _fit_model(x: torch.Tensor, theta: torch.Tensor, target: torch.Tensor,
               candidate_rows: torch.Tensor, config: MomentTrainingConfig,
               kind: str, seed: int):
    """Choose epochs with inner validation, then refit on every candidate row."""
    n_val = max(1, int(len(candidate_rows) * config.validation_fraction))
    if len(candidate_rows) - n_val < 2:
        raise ValueError('Each fold needs at least 2 optimization and 1 validation cubes.')
    permutation = torch.randperm(len(candidate_rows), generator=torch.Generator().manual_seed(seed))
    train_rows = candidate_rows[permutation[n_val:]]
    val_rows = candidate_rows[permutation[:n_val]]
    selection_scaling = _scaling(x, theta, train_rows)
    x_select, y_select = _normalized(x, target, selection_scaling, kind)
    model = _new_model(x_select, y_select, config, kind, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = _loader(x_select, y_select, train_rows, config, seed)
    best_loss, best_epoch, stale = float('inf'), 0, 0
    for epoch in range(config.max_epochs):
        _train_epoch(model, loader, optimizer, config.device)
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((model(x_select[val_rows]) - y_select[val_rows]) ** 2).item()
        if not torch.isfinite(torch.tensor(val_loss)):
            raise ValueError('Moment Network validation loss became non-finite.')
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    # Refit with all candidate rows and their own scaling. For fold models,
    # candidate rows exclude the held-out fold; all models exclude outer test.
    scaling = _scaling(x, theta, candidate_rows)
    x_fit, y_fit = _normalized(x, target, scaling, kind)
    model = _new_model(x_fit, y_fit, config, kind, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = _loader(x_fit, y_fit, candidate_rows, config, seed)
    for _ in range(best_epoch):
        _train_epoch(model, loader, optimizer, config.device)
    model.eval()
    return model, best_epoch, scaling


@dataclass
class MomentFitResult:
    mean_network: nn.Module
    variance_network: nn.Module
    split: DatasetSplit
    training_config: MomentTrainingConfig
    x_mean: torch.Tensor
    x_scale: torch.Tensor
    theta_mean: torch.Tensor
    theta_scale: torch.Tensor
    mean_best_epoch: int
    variance_best_epoch: int
    oof_means: torch.Tensor
    oof_fold: torch.Tensor
    oof_squared_residuals: torch.Tensor
    oof_best_epochs: tuple[int, ...]

    def predict_moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return marginal posterior means and standard deviations in physical units."""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.training_config.device)
        if x.shape[-1] != self.split.data.x.shape[1] or x.ndim not in (1, 2) or not torch.isfinite(x).all():
            raise ValueError('x must be a finite feature vector or batch with the training dimension.')
        with torch.no_grad():
            normalized = (x - self.x_mean) / self.x_scale
            mean = self.mean_network(normalized) * self.theta_scale + self.theta_mean
            variance = self.variance_network(normalized) * self.theta_scale.square()
            std = torch.sqrt(variance.clamp_min(1e-12))
        return mean.cpu(), std.cpu()


class MomentTrainer:
    def __init__(self, config: MomentTrainingConfig | None = None):
        self.config = config or MomentTrainingConfig()

    def fit(self, data: SpectrumDataset) -> MomentFitResult:
        cfg = self.config
        split = data.split(cfg.n_train, cfg.split_seed)
        if cfg.device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError(f'Device {cfg.device!r} requested, but CUDA is unavailable.')
        x = split.x_train.to(cfg.device)
        theta = split.theta_train.to(cfg.device)
        rows = torch.arange(cfg.n_train)
        permutation = torch.randperm(cfg.n_train, generator=torch.Generator().manual_seed(cfg.split_seed))
        folds = torch.tensor_split(permutation, 5)
        oof_means = torch.empty_like(theta)
        oof_fold = torch.full((cfg.n_train,), -1, dtype=torch.int64)
        oof_best_epochs = []
        for fold_id, held_out in enumerate(folds):
            fit_rows = rows[~torch.isin(rows, held_out)]
            fold_model, fold_best_epoch, scaling = _fit_model(
                x, theta, theta, fit_rows, cfg, kind='mean', seed=cfg.train_seed + fold_id + 1)
            x_mean, x_scale, theta_mean, theta_scale = scaling
            with torch.no_grad():
                oof_means[held_out] = (fold_model((x[held_out] - x_mean) / x_scale)
                                       * theta_scale + theta_mean)
            oof_fold[held_out] = fold_id
            oof_best_epochs.append(fold_best_epoch)
        if (oof_fold < 0).any() or not torch.isfinite(oof_means).all():
            raise ValueError('Incomplete out-of-fold predictions.')

        # Residuals are formed in physical parameter units. The variance model
        # normalizes their squares using only the relevant training partition.
        squared_residual = (theta - oof_means).square().detach()
        mean_network, mean_best_epoch, scaling = _fit_model(x, theta, theta, rows, cfg,
                                                             kind='mean', seed=cfg.train_seed)
        variance_network, variance_best_epoch, variance_scaling = _fit_model(
            x, theta, squared_residual, rows, cfg, kind='variance', seed=cfg.train_seed + 6)
        if any(not torch.equal(a, b) for a, b in zip(scaling, variance_scaling)):
            raise ValueError('Final mean and variance normalization must match.')
        return MomentFitResult(mean_network, variance_network, split, cfg, *scaling,
                               mean_best_epoch, variance_best_epoch,
                               oof_means.detach().cpu(), oof_fold.cpu(),
                               squared_residual.cpu(), tuple(oof_best_epochs))
