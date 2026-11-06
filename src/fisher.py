'''Fisher forecasts and score (MOPED-like) compression from the Quijote fixed sets.

Following Kreisch et al. (2022, GIGANTES, Sec. 3) and Villaescusa-Navarro et
al. (2020): the covariance comes from the fiducial realizations, the
derivatives from matched +/- realizations (same initial seeds), and

    F_ab = dmu/dtheta_a  C^-1  dmu/dtheta_b,   sigma_a = sqrt[(F^-1)_aa].

The inverse covariance is debiased with the Hartlap factor (N - D - 2)/(N - 1).
For Mnu the fiducial is ``fiducial_ZA`` and, by default, the second-order
forward difference (-3 P_0 + 4 P_0.1 - P_0.2) / 0.2 is used.

Features are exactly the ones the NPE sees (same blocks, cuts and transforms),
so Fisher errors are the Gaussian-likelihood reference for the NPE widths at
the fiducial cosmology.
'''
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .config import ALL_PARAMETER_LABELS, DERIVATIVE_SETS, FIDUCIAL, MNU_SETS, FisherConfig
from .features import QuijoteSummaryLoader, parse_blocks


@dataclass
class FisherResult:
    parameters: tuple[str, ...]
    blocks: tuple[str, ...]
    fisher: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    derivatives: np.ndarray
    derivative_errors: np.ndarray
    n_covariance: int
    n_derivative: dict[str, int]
    hartlap_factor: float
    condition_number: float
    fiducial: dict[str, float] = field(default_factory=lambda: dict(FIDUCIAL))
    block_sizes: tuple[int, ...] = ()
    fisher_compressed: np.ndarray | None = None

    @property
    def fisher_combined(self) -> np.ndarray | None:
        '''Matrix geometric mean of the standard (optimistic) and compressed (conservative) Fisher.

        Coulton & Wandelt (2023): the two estimators are biased in opposite
        directions by derivative noise; their geometric mean largely cancels it.
        '''
        if self.fisher_compressed is None:
            return None
        return geometric_mean(self.fisher, self.fisher_compressed)

    @property
    def sigma_compressed(self) -> np.ndarray:
        return _sigma(self.fisher_compressed)

    @property
    def sigma_combined(self) -> np.ndarray:
        return _sigma(self.fisher_combined)

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.fisher)

    @property
    def sigma(self) -> np.ndarray:
        '''Marginalized 1-sigma errors.'''
        return np.sqrt(np.diag(self.inverse))

    @property
    def conditional_sigma(self) -> np.ndarray:
        '''Errors with every other parameter fixed.'''
        return 1.0 / np.sqrt(np.diag(self.fisher))

    def sigma_without(self, fixed: tuple[str, ...]) -> np.ndarray:
        keep = [i for i, p in enumerate(self.parameters) if p not in fixed]
        return np.sqrt(np.diag(np.linalg.inv(self.fisher[np.ix_(keep, keep)])))

    def correlation(self) -> np.ndarray:
        d = np.sqrt(np.diag(self.covariance))
        return self.covariance / np.outer(d, d)

    def summary(self):
        import pandas as pd
        theta = np.array([self.fiducial[p] for p in self.parameters])
        return pd.DataFrame({'param': self.parameters, 'fiducial': theta, 'sigma_fisher': self.sigma,
                             'sigma_fisher_compressed': self.sigma_compressed,
                             'sigma_fisher_combined': self.sigma_combined,
                             'sigma_fisher_conditional': self.conditional_sigma,
                             'relative_error': self.sigma / np.where(theta == 0, np.nan, np.abs(theta)),
                             'n_covariance': self.n_covariance,
                             'n_derivative': [self.n_derivative[p] for p in self.parameters],
                             'hartlap_factor': self.hartlap_factor})

    def labels(self):
        return [ALL_PARAMETER_LABELS.get(p, p) for p in self.parameters]


class FisherForecaster:
    def __init__(self, loader: QuijoteSummaryLoader, config: FisherConfig | None = None):
        self.loader = loader
        self.config = config or FisherConfig()
        self._sets: dict = {}

    def _set(self, set_name, blocks):
        key = (set_name, tuple(b.name for b in blocks))
        if key not in self._sets:
            self._sets[key] = self.loader.load_set(set_name, blocks)
        return self._sets[key]

    def compute(self, blocks, n_covariance: int | None = None, n_derivative: int | None = None,
                parameters: tuple[str, ...] | None = None) -> FisherResult:
        blocks = parse_blocks(blocks)
        parameters = tuple(parameters or self.config.parameters)
        n_covariance = n_covariance or self.config.n_covariance
        n_derivative = n_derivative or self.config.n_derivative

        fiducial = self._set('fiducial', blocks)
        x = fiducial.x[:n_covariance] if n_covariance else fiducial.x
        n, d = x.shape
        if n_covariance and n < n_covariance:
            raise ValueError(f'Only {n} fiducial realizations available (< n_covariance={n_covariance}).')
        if d < len(parameters):
            raise ValueError(f'{d} features cannot constrain {len(parameters)} parameters in a Fisher forecast '
                             f'({[b.name for b in blocks]}); add blocks or fix parameters.')
        if n - d - 2 <= 0:
            raise ValueError(f'{d} features need more than {d + 2} fiducial realizations for an invertible '
                             f'covariance (have {n}); cut scales or blocks.')
        mean = x.mean(axis=0)
        covariance = np.cov(x, rowvar=False, ddof=1).reshape(d, d)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues.min() <= eigenvalues.max() * 1e-12:
            raise ValueError('Fiducial covariance is (numerically) singular; some features are degenerate.')
        hartlap = (n - d - 2) / (n - 1) if self.config.hartlap else 1.0
        precision = hartlap * np.linalg.inv(covariance)
        precision = 0.5 * (precision + precision.T)

        derivatives, errors, used, diffs = [], [], {}, []
        for parameter in parameters:
            diff, step = self._finite_differences(parameter, blocks, n_derivative)
            derivatives.append(diff.mean(axis=0) / step)
            errors.append(diff.std(axis=0, ddof=1) / np.sqrt(len(diff)) / abs(step))
            used[parameter] = len(diff)
            diffs.append(diff / step)
        derivatives, errors = np.array(derivatives), np.array(errors)
        fisher = derivatives @ precision @ derivatives.T
        fisher = 0.5 * (fisher + fisher.T)
        if np.linalg.eigvalsh(fisher).min() <= 0:
            raise ValueError('Fisher matrix is not positive definite (parameters degenerate for these blocks).')
        return FisherResult(parameters, tuple(b.name for b in blocks), fisher, mean, covariance, precision,
                            derivatives, errors, n, used, hartlap, float(np.linalg.cond(fisher)),
                            block_sizes=fiducial.block_sizes,
                            fisher_compressed=_compressed_fisher(x, precision, diffs, self.config.compressed_splits,
                                                                 self.config.split_seed))

    def _finite_differences(self, parameter, blocks, n_derivative):
        '''Per-realization differences (already matched by seed) and the step they divide by.'''
        if parameter == 'Mnu':
            sets = {name: self._set(name, blocks) for name, _ in MNU_SETS[:3]}
            ids = _common_ids(*sets.values())[:n_derivative]
            za, p, pp = (sets[name].x[_rows(sets[name], ids)] for name, _ in MNU_SETS[:3])
            step = MNU_SETS[1][1]
            if self.config.mnu_derivative == 'second_order':
                return (-3 * za + 4 * p - pp) / 2.0, step
            return p - za, step
        plus_name, minus_name, plus_value, minus_value = DERIVATIVE_SETS[parameter]
        plus, minus = self._set(plus_name, blocks), self._set(minus_name, blocks)
        ids = _common_ids(plus, minus)[:n_derivative]
        if n_derivative and len(ids) < n_derivative:
            raise ValueError(f'{parameter}: only {len(ids)} matched +/- realizations (< {n_derivative}).')
        return plus.x[_rows(plus, ids)] - minus.x[_rows(minus, ids)], plus_value - minus_value

    def convergence(self, blocks, fractions=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), parameters=None):
        '''GIGANTES App. B: marginal errors vs number of covariance / derivative realizations.'''
        import pandas as pd
        blocks = parse_blocks(blocks)
        full = self.compute(blocks, parameters=parameters)
        rows = []
        for kind in ('covariance', 'derivative'):
            for fraction in fractions:
                if kind == 'covariance':
                    n = int(round(fraction * full.n_covariance))
                    if n - full.mean.size - 2 <= 0:
                        continue
                    result = self.compute(blocks, n_covariance=n, parameters=parameters)
                else:
                    n = int(round(fraction * min(full.n_derivative.values())))
                    if n < 2:
                        continue
                    result = self.compute(blocks, n_derivative=n, parameters=parameters)
                combined = result.sigma_combined if result.fisher_compressed is not None else [np.nan] * len(
                    result.parameters)
                combined_full = full.sigma_combined if full.fisher_compressed is not None else [np.nan] * len(
                    full.parameters)
                for p, s, s_full, c, c_full in zip(result.parameters, result.sigma, full.sigma, combined,
                                                   combined_full):
                    rows.append(dict(kind=kind, n=n, fraction=fraction, param=p, sigma=s, ratio_to_full=s / s_full,
                                     sigma_combined=c, combined_ratio_to_full=c / c_full))
        return pd.DataFrame(rows)


def _sigma(fisher):
    return None if fisher is None else np.sqrt(np.diag(np.linalg.inv(fisher)))


def geometric_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''Matrix geometric mean a # b = a^1/2 (a^-1/2 b a^-1/2)^1/2 a^1/2 of SPD matrices.'''
    def power(m, p):
        w, v = np.linalg.eigh(0.5 * (m + m.T))
        return (v * w ** p) @ v.T
    root, inv_root = power(a, 0.5), power(a, -0.5)
    out = root @ power(inv_root @ b @ inv_root, 0.5) @ root
    return 0.5 * (out + out.T)


def _compressed_fisher(x_fid: np.ndarray, precision: np.ndarray, diffs: list[np.ndarray], n_splits: int = 100,
                       seed: int = 0) -> np.ndarray | None:
    '''Conservative Fisher: compress with one half of the derivative sims, evaluate with the other.

    Averaged over ``n_splits`` random half-splits (both assignments of each).
    One split alone is noisy (~20 % in sigma for Omega_b and h with ~200
    features), enough to make the combined estimator non-monotonic when
    blocks are added. Realizations are permuted jointly for all parameters so
    seed-matched differences stay paired. The compressed statistics
    t = dmu_A C^-1 x have P dimensions, so their covariance (from the fiducial
    sims) and its Hartlap factor are well determined.
    '''
    n = min(len(d) for d in diffs)
    if n < 4:
        return None
    half = n // 2
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(n) for _ in range(n_splits)]
    estimates = []
    for order in orders:
        a, b = order[:half], order[half:2 * half]
        estimates += [_compressed_once(x_fid, precision, diffs, a, b), _compressed_once(x_fid, precision, diffs, b, a)]
    fisher = np.mean(estimates, axis=0)
    return fisher if np.linalg.eigvalsh(fisher).min() > 0 else None


def _compressed_once(x_fid, precision, diffs, first, second):
    '''Compressed Fisher with derivatives from rows ``first`` (compression) and ``second`` (evaluation).'''
    d_first = np.array([d[first].mean(axis=0) for d in diffs])
    d_second = np.array([d[second].mean(axis=0) for d in diffs])
    compression = d_first @ precision                      # (P, D)
    t = x_fid @ compression.T                              # (N, P)
    n_fid, p = t.shape
    cov_t = np.atleast_2d(np.cov(t, rowvar=False, ddof=1))
    dt = d_second @ compression.T                          # dt[a, b] = d t_b / d theta_a
    fisher = dt @ (((n_fid - p - 2) / (n_fid - 1)) * np.linalg.inv(cov_t)) @ dt.T
    return 0.5 * (fisher + fisher.T)


def _common_ids(*sets):
    ids = sets[0].ids
    for s in sets[1:]:
        ids = np.intersect1d(ids, s.ids)
    if not len(ids):
        raise ValueError('No matched realizations between finite-difference sets.')
    return ids


def _rows(data, ids):
    position = {int(r): i for i, r in enumerate(data.ids)}
    return np.array([position[int(r)] for r in ids])


@dataclass
class ScoreCompressor:
    '''Linear score compression t = theta_fid + F^-1 dmu^T C^-1 (x - mu_fid).

    The output is the quasi-maximum-likelihood estimate of the parameters
    (same units as theta) and keeps all the Fisher information at the fiducial
    point. It is built from fixed-cosmology sets only, so it never sees the
    latin-hypercube test cubes.
    '''
    mean: np.ndarray
    matrix: np.ndarray
    offset: np.ndarray
    parameters: tuple[str, ...]
    blocks: tuple[str, ...]

    @classmethod
    def from_fisher(cls, result: FisherResult) -> 'ScoreCompressor':
        matrix = np.linalg.solve(result.fisher, result.derivatives @ result.precision)  # (P, D)
        offset = np.array([result.fiducial[p] for p in result.parameters])
        return cls(result.mean, matrix, offset, result.parameters, result.blocks)

    @property
    def input_dim(self) -> int:
        return self.mean.size

    @property
    def output_dim(self) -> int:
        return len(self.parameters)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x)
        mean = torch.as_tensor(self.mean, dtype=torch.float64, device=x.device)
        matrix = torch.as_tensor(self.matrix, dtype=torch.float64, device=x.device)
        offset = torch.as_tensor(self.offset, dtype=torch.float64, device=x.device)
        return ((x.to(torch.float64) - mean) @ matrix.T + offset).to(torch.float32)
