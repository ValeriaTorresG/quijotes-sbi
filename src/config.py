from dataclasses import dataclass
from math import isfinite
from numbers import Integral

PARAMETER_NAMES = ('Omega_m', 'Omega_b', 'h', 'n_s', 'sigma8')
PARAMETER_LABELS = (r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$')
ENVIRONMENT_FILES = {env: f'group_003_pk_halo_{env}_real_los2_CIC_N512_dk2kf_kmax0p5_mono.npz'
                     for env in ('all', 'void', 'sheet', 'filament', 'knot')}
ENVIRONMENT_FILES['random_void'] = ('group_003_pk_random_void_real_los2_CIC_N512_dk2kf_kmax0p5_mono.npz')


@dataclass(frozen=True)
class SpectrumConfig:
    '''Inclusive k range; use transform='none' to retain raw Pk values'''
    kmin: float = 0.01
    kmax: float = 0.5
    transform: str = 'log10'
    grid_rtol: float = 1e-6
    grid_atol: float = 1e-10

    def __post_init__(self):
        if not (isfinite(self.kmin) and isfinite(self.kmax) and 0 <= self.kmin <= self.kmax):
            raise ValueError('Require finite 0 <= kmin <= kmax.')

        if self.transform not in ('log10', 'none'):
            raise ValueError('-----> transform must be log10 or none')

        if any(not isfinite(v) or v < 0 for v in (self.grid_rtol, self.grid_atol)):
            raise ValueError('Grid tolerances must be finite and nonnegative.')


@dataclass(frozen=True)
class PriorConfig:
    '''Uniform Latin-hypercube prior, in the parameter-file column order.'''
    low: tuple[float, ...] = (0.1, 0.03, 0.5, 0.8, 0.6)
    high: tuple[float, ...] = (0.5, 0.07, 0.9, 1.2, 1.0)

    def __post_init__(self):
        if not self.low or len(self.low) != len(self.high):
            raise ValueError('Prior bounds must have matching, nonempty dimensions.')

        if any(not (isfinite(lo) and isfinite(hi) and lo < hi)
               for lo, hi in zip(self.low, self.high)):
            raise ValueError('Each prior bound must be finite with low < high.')


@dataclass(frozen=True)
class TrainingConfig:
    '''NSF and training defaults from sbi_env.ipynb (CPU by default).'''
    n_train: int = 720
    split_seed: int = 42
    train_seed: int = 42
    hidden_features: int = 128
    num_transforms: int = 5
    batch_size: int = 128
    learning_rate: float = 5e-4
    validation_fraction: float = 0.1
    patience: int = 30
    max_epochs: int = 1000
    device: str = 'cpu'
    show_train_summary: bool = True

    def __post_init__(self):
        for name in ('n_train', 'hidden_features', 'num_transforms', 'batch_size', 'patience', 'max_epochs'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')

        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive.')

        if not 0 < self.validation_fraction < 1:
            raise ValueError('validation_fraction must lie between 0 and 1.')

        for name in ('split_seed', 'train_seed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer.')


@dataclass(frozen=True)
class EvaluationConfig:
    n_samples: int = 5000
    sample_seed: int = 123
    show_progress: bool = True

    def __post_init__(self):
        if isinstance(self.n_samples, bool) or not isinstance(self.n_samples, Integral) or self.n_samples < 2:
            raise ValueError('n_samples must be an integer >= 2 to estimate sample std.')

        if isinstance(self.sample_seed, bool) or not isinstance(self.sample_seed, Integral) or self.sample_seed < 0:
            raise ValueError('sample_seed must be a nonnegative integer.')