'''Posterior validation.

For every held-out cube (or fiducial realization) we store posterior means,
standard deviations, 68/95 % central intervals, the marginal PIT value
u = P(theta < theta_true | x) (simulation-based calibration ranks), the joint
log density at the truth, and TARP statistics (Lemos et al. 2023).

Interpretation
--------------
* ``coverage_68/95`` and the calibration curve: fraction of truths inside the
  central credible interval vs its nominal level (marginal calibration).
* ``pit_ks_pvalue``: KS test of PIT uniformity (SBC); small values flag
  over-/under-confidence or bias.
* ``chi2``: mean of ((mean - truth) / std)^2; ~1 for Gaussian-calibrated posteriors.
* ``tarp_max_deviation``: max |ECP(alpha) - alpha| of the joint TARP test;
  sensitive to miscalibration of the joint posterior, not only marginals.
* ``nll``: mean -log q(theta_true | x); proper score for comparing models on
  the same test cubes (lower is better).
* ``*_interior``: metrics restricted to truths away from the prior edges,
  where the boundary truncation does not dominate.
* ``std_over_prior``: posterior std / prior std (1 = no information).
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy import stats
from tqdm.auto import tqdm

from .config import EvaluationConfig, PriorConfig
from .inference import FitResult, TrainedPosterior

if TYPE_CHECKING:
    import pandas as pd

LEVELS = np.round(np.linspace(0.05, 0.95, 19), 2)
_Z68, _Z95 = 0.994457883209753, 1.95996398454005


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
    interval_method: str = 'posterior_quantiles'
    pit: np.ndarray | None = None
    log_prob: np.ndarray | None = None
    tarp_f: np.ndarray | None = None
    acceptance: np.ndarray | None = None
    prior_low: np.ndarray | None = None
    prior_high: np.ndarray | None = None

    def _interior(self) -> np.ndarray:
        if self.prior_low is None:
            return np.ones_like(self.truth, dtype=bool)
        width = self.prior_high - self.prior_low
        margin = self.config.interior_margin * width
        return (self.truth >= self.prior_low + margin) & (self.truth <= self.prior_high - margin)

    @property
    def summary(self) -> pd.DataFrame:
        import pandas as pd

        errors = self.means - self.truth
        inside68 = (self.truth >= self.low68) & (self.truth <= self.high68)
        inside95 = (self.truth >= self.low95) & (self.truth <= self.high95)
        interior = self._interior()
        variance = self.truth.var(axis=0)
        columns = {'param': self.parameter_names,
                   'bias': errors.mean(axis=0),
                   'RMSE': np.sqrt((errors ** 2).mean(axis=0)),
                   'R2': 1 - (errors ** 2).mean(axis=0) / np.where(variance > 0, variance, np.nan),
                   'mean_posterior_std': self.stds.mean(axis=0),
                   'mean_68_interval_width': (self.high68 - self.low68).mean(axis=0),
                   'coverage_68': inside68.mean(axis=0),
                   'coverage_95': inside95.mean(axis=0),
                   'chi2': ((errors / self.stds) ** 2).mean(axis=0)}
        if self.prior_low is not None:
            columns['std_over_prior'] = self.stds.mean(axis=0) / ((self.prior_high - self.prior_low) / np.sqrt(12))
        if self.pit is not None:
            columns['pit_ks_pvalue'] = [stats.kstest(self.pit[:, j], 'uniform').pvalue
                                        for j in range(self.pit.shape[1])]
            curve = self.coverage_curve()
            columns['calibration_error'] = (curve.assign(err=(curve.empirical - curve.level).abs())
                                            .groupby('param', sort=False).err.mean()
                                            .reindex(self.parameter_names).to_numpy())
        with np.errstate(invalid='ignore'):
            columns['RMSE_interior'] = [np.sqrt((errors[m, j] ** 2).mean()) if m.any() else np.nan
                                        for j, m in enumerate(interior.T)]
            columns['coverage_68_interior'] = [inside68[m, j].mean() if m.any() else np.nan
                                               for j, m in enumerate(interior.T)]
        columns['n'] = len(self.truth)
        return pd.DataFrame(columns)

    @property
    def scores(self) -> dict:
        out = {'n': int(len(self.truth))}
        if self.log_prob is not None:
            finite = np.isfinite(self.log_prob)
            out['nll'] = float(-self.log_prob[finite].mean()) if finite.any() else float('nan')
            out['n_logprob_nonfinite'] = int((~finite).sum())
        if self.tarp_f is not None:
            alpha, ecp = self.tarp_curve()
            out['tarp_max_deviation'] = float(np.abs(ecp - alpha).max())
            out['tarp_ks_pvalue'] = float(stats.kstest(self.tarp_f.ravel(), 'uniform').pvalue)
        if self.acceptance is not None:
            out['mean_prior_leakage'] = float(1 - self.acceptance.mean())
        return out

    def coverage_curve(self, levels=LEVELS) -> pd.DataFrame:
        '''Empirical coverage of central credible intervals vs nominal level (from PIT).'''
        import pandas as pd
        if self.pit is None:
            raise ValueError('No PIT values stored.')
        n = len(self.pit)
        rows = []
        for j, name in enumerate(self.parameter_names):
            distance = np.abs(self.pit[:, j] - 0.5)
            for level in levels:
                k = int((distance <= level / 2 + 1e-12).sum())
                low, high = stats.beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)  # Jeffreys interval
                rows.append(dict(param=name, level=float(level), empirical=k / n, ci_low=low, ci_high=high))
        return pd.DataFrame(rows)

    def tarp_curve(self, n_alpha: int = 101):
        alpha = np.linspace(0, 1, n_alpha)
        f = self.tarp_f.ravel()
        return alpha, np.array([(f < a).mean() for a in alpha])

    def arrays(self) -> dict[str, np.ndarray]:
        out = dict(truth=self.truth, means=self.means, stds=self.stds, low68=self.low68, high68=self.high68,
                   low95=self.low95, high95=self.high95, cube_ids=self.cube_ids)
        for name in ('pit', 'log_prob', 'tarp_f', 'acceptance'):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        return out


def _prior_arrays(prior: PriorConfig | None):
    if prior is None:
        return None, None
    return np.asarray(prior.low, dtype=float), np.asarray(prior.high, dtype=float)


def evaluate_samples(model: TrainedPosterior, x: torch.Tensor, truth: np.ndarray, ids: np.ndarray,
                     config: EvaluationConfig, desc: str = 'evaluate') -> EvaluationResult:
    '''Sample-based evaluation of an NPE posterior on paired (x, truth) rows.'''
    truth = np.asarray(truth, dtype=float)
    n, n_params = truth.shape
    low, high = _prior_arrays(model.prior)
    rng = np.random.default_rng(config.sample_seed)
    references = rng.uniform(size=(n, config.n_tarp_references, n_params))
    moments = np.empty((2, n, n_params))
    intervals = np.empty((4, n, n_params))
    pit = np.empty((n, n_params))
    tarp_f = np.empty((n, config.n_tarp_references))
    log_prob = np.empty(n)
    acceptance = np.empty(n)
    batches = range(0, n, config.batch_size)
    for b, start in enumerate(tqdm(batches, desc=desc, disable=not config.show_progress)):
        rows = slice(start, min(start + config.batch_size, n))
        xb = x[rows]
        samples = model.sample_batched(xb, config.n_samples, seed=config.sample_seed + b).numpy()  # (S, B, P)
        if samples.shape != (config.n_samples, len(xb), n_params) or not np.isfinite(samples).all():
            raise ValueError(f'Invalid posterior samples for cubes {ids[rows][:5].tolist()}.')
        tb = truth[rows]
        moments[0, rows] = samples.mean(axis=0)
        moments[1, rows] = samples.std(axis=0, ddof=1)
        intervals[:, rows] = np.quantile(samples, [0.025, 0.16, 0.84, 0.975], axis=0)
        pit[rows] = (samples < tb[None]).mean(axis=0)
        # TARP in prior-normalized coordinates.
        s_norm = (samples - low) / (high - low)
        t_norm = (tb - low) / (high - low)
        for r in range(config.n_tarp_references):
            ref = references[rows, r]
            d_true = np.linalg.norm(t_norm - ref, axis=-1)
            d_samp = np.linalg.norm(s_norm - ref[None], axis=-1)
            tarp_f[rows, r] = (d_samp < d_true[None]).mean(axis=0)
        theta_b = torch.as_tensor(tb, dtype=torch.float32)
        accepted = model.acceptance(xb, config.leakage_samples) if config.leakage_samples else None
        log_prob[rows] = model.log_prob(theta_b, xb, leakage_samples=config.leakage_samples,
                                        acceptance=accepted).numpy()
        acceptance[rows] = accepted.mean(0).numpy() if accepted is not None else np.nan
    low95, low68, high68, high95 = intervals
    return EvaluationResult(truth=truth, means=moments[0], stds=moments[1], low68=low68, high68=high68,
                            low95=low95, high95=high95, cube_ids=np.asarray(ids).copy(),
                            parameter_names=model.parameter_names, config=config, pit=pit, log_prob=log_prob,
                            tarp_f=tarp_f, acceptance=None if np.isnan(acceptance).all() else acceptance,
                            prior_low=low, prior_high=high)


def evaluate_gaussian(means: np.ndarray, stds: np.ndarray, truth: np.ndarray, ids: np.ndarray,
                      parameter_names, config: EvaluationConfig, prior: PriorConfig | None = None) -> EvaluationResult:
    '''Evaluation for Moment Networks: independent Gaussian marginals from (mean, std).'''
    low, high = _prior_arrays(prior)
    z = (truth - means) / stds
    log_prob = stats.norm.logpdf(z).sum(axis=1) - np.log(stds).sum(axis=1)
    return EvaluationResult(truth=truth, means=means, stds=stds, low68=means - _Z68 * stds,
                            high68=means + _Z68 * stds, low95=means - _Z95 * stds, high95=means + _Z95 * stds,
                            cube_ids=np.asarray(ids).copy(), parameter_names=tuple(parameter_names), config=config,
                            interval_method='gaussian_from_moments', pit=stats.norm.cdf(z), log_prob=log_prob,
                            prior_low=low, prior_high=high)


class PosteriorEvaluator:
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def evaluate(self, fit: FitResult) -> EvaluationResult:
        return evaluate_samples(fit.model, fit.split.x_test, fit.split.theta_test.cpu().numpy(),
                                fit.split.test_cube_ids, self.config, desc='test')

    def evaluate_fixed(self, fit: FitResult, x: torch.Tensor, theta: np.ndarray, ids: np.ndarray,
                       desc: str = 'fiducial') -> EvaluationResult:
        truth = np.broadcast_to(np.asarray(theta, dtype=float), (len(ids), len(theta))).copy()
        return evaluate_samples(fit.model, x, truth, ids, self.config, desc=desc)


class MomentEvaluator:
    """Evaluate direct moments; central intervals assume Gaussian marginal shapes."""

    def __init__(self, config: EvaluationConfig | None = None, prior: PriorConfig | None = None):
        self.config = config or EvaluationConfig()
        self.prior = prior or PriorConfig()

    def _evaluate(self, fit, x, truth, ids) -> EvaluationResult:
        means, stds = fit.predict_moments(x)
        means, stds = means.numpy().astype(float), stds.numpy().astype(float)
        if means.shape != truth.shape or stds.shape != truth.shape or not np.isfinite(means).all() \
                or not np.isfinite(stds).all():
            raise ValueError('Invalid Moment Network predictions.')
        prior = self.prior if len(self.prior.low) == truth.shape[1] else None
        return evaluate_gaussian(means, stds, truth, ids, fit.split.data.parameter_names, self.config, prior)

    def evaluate(self, fit) -> EvaluationResult:
        return self._evaluate(fit, fit.split.x_test, fit.split.theta_test.detach().cpu().numpy().astype(float),
                              fit.split.test_cube_ids)

    def evaluate_fixed(self, fit, x, theta, ids, desc: str = 'fiducial') -> EvaluationResult:
        truth = np.broadcast_to(np.asarray(theta, dtype=float), (len(ids), len(theta))).copy()
        return self._evaluate(fit, x, truth, ids)
