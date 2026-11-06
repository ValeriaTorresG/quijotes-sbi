'''Constants and validated configuration objects.

Parameter order everywhere is ``PARAMETER_NAMES`` (the column order of
``latin_hypercube_params.txt``). ``Mnu`` only exists in the Fisher forecasts:
the Quijote latin hypercube has massless neutrinos, so NPE infers five
parameters.
'''
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral
from pathlib import Path

PARAMETER_NAMES = ('Omega_m', 'Omega_b', 'h', 'n_s', 'sigma8')
PARAMETER_LABELS = (r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$')
ALL_PARAMETER_LABELS = dict(zip(PARAMETER_NAMES, PARAMETER_LABELS)) | {'Mnu': r'$M_\nu$'}

ENVIRONMENTS = ('void', 'sheet', 'filament', 'knot')
HALO_SAMPLES = ('all',) + ENVIRONMENTS
PK_SAMPLES = HALO_SAMPLES + ('random_void',)

# Legacy file names used by QuijoteDataLoader (latin hypercube, real space).
ENVIRONMENT_FILES = {env: f'group_003_pk_halo_{env}_real_los2_CIC_N512_dk2kf_kmax0p5_mono.npz'
                     for env in HALO_SAMPLES}
ENVIRONMENT_FILES['random_void'] = 'group_003_pk_random_void_real_los2_CIC_N512_dk2kf_kmax0p5_mono.npz'

# Quijote fiducial cosmology and finite-difference sets (Villaescusa-Navarro
# et al. 2020, Table 1). ``fiducial_ZA`` is the Zel'dovich fiducial paired with
# the massive-neutrino sets.
FIDUCIAL = {'Omega_m': 0.3175, 'Omega_b': 0.049, 'h': 0.6711, 'n_s': 0.9624, 'sigma8': 0.834, 'Mnu': 0.0}
DERIVATIVE_SETS = {
    'Omega_m': ('Om_p', 'Om_m', 0.3275, 0.3075),
    'Omega_b': ('Ob2_p', 'Ob2_m', 0.051, 0.047),
    'h': ('h_p', 'h_m', 0.6911, 0.6511),
    'n_s': ('ns_p', 'ns_m', 0.9824, 0.9424),
    'sigma8': ('s8_p', 's8_m', 0.849, 0.819),
}
MNU_SETS = (('fiducial_ZA', 0.0), ('Mnu_p', 0.1), ('Mnu_pp', 0.2), ('Mnu_ppp', 0.4))
FIXED_SETS = ('fiducial', 'fiducial_ZA') + tuple(s for p, m, *_ in DERIVATIVE_SETS.values() for s in (p, m)) \
    + tuple(name for name, _ in MNU_SETS[1:])
LATIN_HYPERCUBE = 'latin_hypercube'
ALL_SETS = (LATIN_HYPERCUBE,) + FIXED_SETS


def set_cosmology(set_name: str) -> dict[str, float]:
    '''Cosmology of a fixed-parameter Quijote set, including Mnu.'''
    if set_name in ('fiducial', 'fiducial_ZA'):
        return dict(FIDUCIAL)
    for parameter, (plus, minus, value_plus, value_minus) in DERIVATIVE_SETS.items():
        if set_name in (plus, minus):
            return FIDUCIAL | {parameter: value_plus if set_name == plus else value_minus}
    for name, mnu in MNU_SETS:
        if set_name == name:
            return FIDUCIAL | {'Mnu': mnu}
    raise ValueError(f'{set_name!r} is not a fixed-cosmology Quijote set; known: {FIXED_SETS}')


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')


def _seed(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer.')


@dataclass(frozen=True)
class DataLayout:
    '''Where each product lives. ``{set}``, ``{id}`` and ``{sample}`` are filled in.

    Real-space P(k) for the latin hypercube is in ``PowerSpectrum`` with the
    ``_los2_..._mono`` suffix; for the fiducial and derivative sets it is in
    ``PowerSpectrumReal``. Both were checked to be bit-identical where they overlap.
    '''
    root: Path = Path('/pscratch/sd/v/vtorresg/quijotes')
    params_file: Path = Path(__file__).resolve().parents[1] / 'latin_hypercube_params.txt'
    pk_real_lh: str = 'PowerSpectrum/FoF/{set}/{id}/group_003_pk_{sample}_real_los2_CIC_N512_dk2kf_kmax0p5_mono.npz'
    pk_real_fixed: str = 'PowerSpectrumReal/FoF/{set}/{id}/group_003_pk_{sample}_real_CIC_N512_dk2kf_kmax0p5.npz'
    pk_rsd: str = 'PowerSpectrum/FoF/{set}/{id}/group_003_pk_{sample}_rsd2_CIC_N512_dk2kf_kmax0p5.npz'
    astra: str = 'ASTRA/FoF/{set}/{id}/group_003_probability.fits.gz'
    halos: str = 'Halos/FoF/{set}/{id}/groups_003/group_tab_003.0'
    snapshot: int = 3
    redshift: float = 0.5
    box_size: float = 1000.0
    n_particles_side: int = 512
    min_halo_particles: int = 20
    expected_realizations: tuple[tuple[str, int], ...] = ((LATIN_HYPERCUBE, 2000), ('fiducial', 2000))
    default_fixed_realizations: int = 500

    def __post_init__(self):
        object.__setattr__(self, 'root', Path(self.root))
        object.__setattr__(self, 'params_file', Path(self.params_file))

    def n_expected(self, set_name: str) -> int:
        return dict(self.expected_realizations).get(set_name, self.default_fixed_realizations)

    def path(self, product: str, set_name: str, realization: int, sample: str | None = None) -> Path:
        if product == 'pk':
            template = self.pk_real_lh if set_name == LATIN_HYPERCUBE else self.pk_real_fixed
        else:
            template = getattr(self, product)
        if sample is not None:
            sample = sample if sample == 'random_void' else f'halo_{sample}'
        return self.root / template.format(set=set_name, id=realization, sample=sample)


@dataclass(frozen=True)
class SpectrumConfig:
    '''Inclusive k range on ``k_center`` and the P(k) feature transform.

    ``shot_noise`` selects ``Pk0_shot_subtracted`` (default) or ``Pk0_raw``
    explicitly. The ambiguous ``Pk0`` key is never used by the new loader: it is
    raw in the latin-hypercube files but shot-noise subtracted in
    ``PowerSpectrumReal``.

    ``transform``: ``asinh`` (default) maps P to asinh(P / asinh_scale), which is
    log-like for P >> asinh_scale and linear around zero, so it accepts the
    negative shot-noise-subtracted values of sparse samples (void halos are
    below shot noise at high k in most cubes). ``log10`` needs P > 0 (use it with
    ``shot_noise='raw'``); ``none`` keeps P unchanged.
    '''
    kmin: float = 0.01
    kmax: float = 0.5
    transform: str = 'asinh'
    grid_rtol: float = 1e-6
    grid_atol: float = 1e-10
    shot_noise: str = 'subtracted'
    asinh_scale: float = 1000.0

    def __post_init__(self):
        if not (isfinite(self.kmin) and isfinite(self.kmax) and 0 <= self.kmin <= self.kmax):
            raise ValueError('Require finite 0 <= kmin <= kmax.')
        if self.transform not in ('asinh', 'log10', 'none'):
            raise ValueError('transform must be asinh, log10 or none.')
        if not isfinite(self.asinh_scale) or self.asinh_scale <= 0:
            raise ValueError('asinh_scale must be finite and positive.')
        if any(not isfinite(v) or v < 0 for v in (self.grid_rtol, self.grid_atol)):
            raise ValueError('Grid tolerances must be finite and nonnegative.')
        if self.shot_noise not in ('raw', 'subtracted'):
            raise ValueError("shot_noise must be 'raw' or 'subtracted'.")


@dataclass(frozen=True)
class FeatureConfig:
    '''How raw products become features (applied identically to every set).'''
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    rsd_multipoles: tuple[int, ...] = (0, 2)
    astra_bins: int = 10
    # 13.4 is above the 20-particle FoF threshold of every latin-hypercube cube
    # (10^13.32 Msun/h at Omega_m = 0.5); the last bin collects all heavier halos.
    hmf_log10_mass_edges: tuple[float, ...] = (13.4, 13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8, 17.0)
    allow_incomplete_hmf: bool = False

    def __post_init__(self):
        if not self.rsd_multipoles or any(ell not in (0, 2, 4) for ell in self.rsd_multipoles) \
                or len(set(self.rsd_multipoles)) != len(self.rsd_multipoles):
            raise ValueError('rsd_multipoles must be distinct values among 0, 2, 4.')
        _positive_int(self.astra_bins, 'astra_bins', 2)
        edges = tuple(float(e) for e in self.hmf_log10_mass_edges)
        if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])) or not all(map(isfinite, edges)):
            raise ValueError('hmf_log10_mass_edges must be finite and strictly increasing.')
        object.__setattr__(self, 'hmf_log10_mass_edges', edges)


@dataclass(frozen=True)
class PriorConfig:
    '''Uniform Latin-hypercube prior, in the parameter-file column order.'''
    low: tuple[float, ...] = (0.1, 0.03, 0.5, 0.8, 0.6)
    high: tuple[float, ...] = (0.5, 0.07, 0.9, 1.2, 1.0)

    def __post_init__(self):
        if not self.low or len(self.low) != len(self.high):
            raise ValueError('Prior bounds must have matching, nonempty dimensions.')
        if any(not (isfinite(lo) and isfinite(hi) and lo < hi) for lo, hi in zip(self.low, self.high)):
            raise ValueError('Each prior bound must be finite with low < high.')


@dataclass(frozen=True)
class TrainingConfig:
    '''NPE settings. ``ensemble_size`` independent flows are mixed with equal weights.

    ``embedding_dim=0`` conditions the flow directly on the z-scored features;
    a positive value adds a jointly trained MLP embedding (useful for long
    concatenated vectors). ``compression='moped'`` replaces the features by the
    score-compressed summaries built from the fiducial and derivative sets.
    '''
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
    ensemble_size: int = 1
    density_estimator: str = 'nsf'
    embedding_dim: int = 0
    embedding_hidden: int = 128
    embedding_layers: int = 2
    compression: str = 'none'
    clip_max_norm: float = 5.0

    def __post_init__(self):
        for name in ('n_train', 'hidden_features', 'num_transforms', 'batch_size', 'patience', 'max_epochs',
                     'ensemble_size', 'embedding_hidden', 'embedding_layers'):
            _positive_int(getattr(self, name), name)
        _positive_int(self.embedding_dim, 'embedding_dim', 0)
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive.')
        if not 0 < self.validation_fraction < 1:
            raise ValueError('validation_fraction must lie between 0 and 1.')
        for name in ('split_seed', 'train_seed'):
            _seed(getattr(self, name), name)
        if self.density_estimator not in ('nsf', 'maf', 'mdn'):
            raise ValueError('density_estimator must be nsf, maf or mdn.')
        if self.compression not in ('none', 'moped'):
            raise ValueError("compression must be 'none' or 'moped'.")
        if not isfinite(self.clip_max_norm) or self.clip_max_norm <= 0:
            raise ValueError('clip_max_norm must be finite and positive.')


@dataclass(frozen=True)
class MomentTrainingConfig:
    """Two-stage Moment Network defaults; each network has its own early stopping."""
    n_train: int = 720
    split_seed: int = 42
    train_seed: int = 42
    hidden_features: int = 128
    hidden_layers: int = 2
    batch_size: int = 128
    learning_rate: float = 5e-4
    validation_fraction: float = 0.1
    patience: int = 30
    max_epochs: int = 1000
    device: str = 'cpu'

    def __post_init__(self):
        if isinstance(self.n_train, bool) or not isinstance(self.n_train, Integral) or self.n_train < 5:
            raise ValueError('Moment Networks require at least 5 training cubes for five folds.')
        for name in ('hidden_features', 'hidden_layers', 'batch_size', 'patience', 'max_epochs'):
            _positive_int(getattr(self, name), name)
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive.')
        if not 0 < self.validation_fraction < 1:
            raise ValueError('validation_fraction must lie between 0 and 1.')
        for name in ('split_seed', 'train_seed'):
            _seed(getattr(self, name), name)


@dataclass(frozen=True)
class EvaluationConfig:
    '''``interior_margin`` is the fraction of each prior width treated as "edge".'''
    n_samples: int = 5000
    sample_seed: int = 123
    show_progress: bool = True
    n_tarp_references: int = 1
    leakage_samples: int = 2000
    interior_margin: float = 0.1
    batch_size: int = 50

    def __post_init__(self):
        if isinstance(self.n_samples, bool) or not isinstance(self.n_samples, Integral) or self.n_samples < 2:
            raise ValueError('n_samples must be an integer >= 2 to estimate sample std.')
        _seed(self.sample_seed, 'sample_seed')
        _positive_int(self.n_tarp_references, 'n_tarp_references')
        _positive_int(self.leakage_samples, 'leakage_samples', 0)
        _positive_int(self.batch_size, 'batch_size')
        if not 0 <= self.interior_margin < 0.5:
            raise ValueError('interior_margin must lie in [0, 0.5).')


@dataclass(frozen=True)
class FisherConfig:
    '''Fisher forecast from the fiducial covariance and finite-difference means.'''
    parameters: tuple[str, ...] = PARAMETER_NAMES
    n_covariance: int | None = None
    n_derivative: int | None = None
    hartlap: bool = True
    mnu_derivative: str = 'second_order'
    # Random half-splits of the derivative sims averaged in the compressed
    # (conservative) Fisher; a single split scatters by ~20 % for long vectors.
    compressed_splits: int = 100
    split_seed: int = 0

    def __post_init__(self):
        known = set(PARAMETER_NAMES) | {'Mnu'}
        _positive_int(self.compressed_splits, 'compressed_splits')
        _seed(self.split_seed, 'split_seed')
        if not self.parameters or len(set(self.parameters)) != len(self.parameters) \
                or not set(self.parameters) <= known:
            raise ValueError(f'parameters must be distinct names among {sorted(known)}.')
        for name in ('n_covariance', 'n_derivative'):
            value = getattr(self, name)
            if value is not None:
                _positive_int(value, name, 2)
        if self.mnu_derivative not in ('second_order', 'first_order'):
            raise ValueError("mnu_derivative must be 'second_order' or 'first_order'.")
