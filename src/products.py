'''Readers for one realization of each raw Quijote/ASTRA product, with validation.

Every reader checks the file's own metadata (set name, realization, snapshot,
box, grid, sample, space) against what the caller expects, so a file copied
into the wrong directory is rejected instead of silently becoming training
data. Readers return small dictionaries of arrays that ``store.ProductStore``
stacks and caches.

Products
--------
``pk``      real-space monopole (raw and shot-noise subtracted) per sample.
``pk_rsd``  redshift-space multipoles (l = 0, 2, 4) per halo sample.
``astra``   per-environment histograms of the ASTRA classification probability.
``hmf``     FoF halo-mass histograms for all halos and per ASTRA environment.
'''
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ENVIRONMENTS, HALO_SAMPLES, LATIN_HYPERCUBE, DataLayout, set_cosmology
from .fits import read_fits_table

PRODUCTS = ('pk', 'pk_rsd', 'astra', 'hmf')
READER_VERSION = 1
# Fine log10(M [Msun/h]) grid stored in the cache; user bins must align with it.
HMF_FINE_EDGES = np.round(np.arange(11.5, 17.0 + 1e-9, 0.05), 6)
RHO_CRIT = 27.7536627  # 1e10 Msun/h per (Mpc/h)^3


class DataValidationError(ValueError):
    '''Raised when a file contradicts the metadata the pipeline expects.'''


def _scalar(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    return value.item() if np.ndim(value) == 0 else value


def _require(condition: bool, message: str, path: Path):
    if not condition:
        raise DataValidationError(f'{message}: {path}')


def _check_common(data, path: Path, layout: DataLayout, set_name: str, realization: int,
                  sample: str, space: str, expected_omega_m: float | None):
    label = _scalar(data, 'parameter', _scalar(data, 'cosmology'))
    _require(label is None or str(label) == set_name, f'File belongs to set {label!r}, not {set_name!r}', path)
    _require(_scalar(data, 'realization', realization) == realization,
             f'File realization {_scalar(data, "realization")} != directory {realization}', path)
    _require(_scalar(data, 'snapshot', layout.snapshot) == layout.snapshot, 'Unexpected snapshot', path)
    _require(abs(float(_scalar(data, 'box_size', layout.box_size)) - layout.box_size) < 1e-6, 'Unexpected box size', path)
    _require(str(_scalar(data, 'sample', sample)) == sample, f'Expected sample {sample!r}', path)
    _require(str(_scalar(data, 'space', space)) == space, f'Expected {space}-space spectrum', path)
    _require(str(_scalar(data, 'mas', 'CIC')) == 'CIC', 'Expected CIC mass assignment', path)
    redshift = _scalar(data, 'redshift')
    _require(redshift is None or abs(float(redshift) - layout.redshift) < 1e-6, 'Unexpected redshift', path)
    omega_m = _scalar(data, 'omega_m')
    if omega_m is not None and expected_omega_m is not None:
        _require(abs(float(omega_m) - expected_omega_m) < 1e-5,
                 f'File Omega_m={float(omega_m)} but parameters give {expected_omega_m}', path)


def _grid(data, path: Path):
    k = np.asarray(data['k_center'], dtype=float)
    low, high = np.asarray(data['k_bin_low'], dtype=float), np.asarray(data['k_bin_high'], dtype=float)
    _require(k.ndim == 1 and k.size > 0 and low.shape == k.shape == high.shape, 'Malformed k grid', path)
    _require(np.isfinite(k).all() and np.all(np.diff(k) > 0) and np.all(low < high)
             and np.all((k >= low) & (k <= high)), 'Invalid k grid', path)
    return k, low, high


def _shot_noise_checks(data, raw: np.ndarray, path: Path, layout: DataLayout):
    n_objects = int(_scalar(data, 'n_objects'))
    shot = float(_scalar(data, 'shot_noise'))
    _require(n_objects > 0, 'n_objects must be positive', path)
    _require(np.isclose(shot, layout.box_size ** 3 / n_objects, rtol=1e-6), 'shot_noise != V / n_objects', path)
    sub = np.asarray(data['Pk0_shot_subtracted'], dtype=float)
    _require(sub.shape == raw.shape and np.allclose(sub, raw - shot, rtol=1e-8, atol=1e-6 * np.abs(raw).max()),
             'Pk0_shot_subtracted != Pk0_raw - shot_noise', path)
    return n_objects, shot, sub


def read_pk(path: Path, layout: DataLayout, set_name: str, realization: int, sample: str,
            expected_omega_m: float | None) -> dict:
    with np.load(path, allow_pickle=False) as data:
        name = sample if sample == 'random_void' else f'halo_{sample}'
        _check_common(data, path, layout, set_name, realization, name, 'real', expected_omega_m)
        k, low, high = _grid(data, path)
        raw = np.asarray(data['Pk0_raw'], dtype=float)
        _require(raw.shape == k.shape and np.isfinite(raw).all(), 'Pk0_raw must be finite on the k grid', path)
        n_objects, shot, sub = _shot_noise_checks(data, raw, path, layout)
        return dict(k=k, k_low=low, k_high=high, pk_raw=raw, pk_sub=sub,
                    n_objects=np.int64(n_objects), shot_noise=np.float64(shot))


def read_pk_rsd(path: Path, layout: DataLayout, set_name: str, realization: int, sample: str,
                expected_omega_m: float | None) -> dict:
    with np.load(path, allow_pickle=False) as data:
        _check_common(data, path, layout, set_name, realization, f'halo_{sample}', 'redshift', expected_omega_m)
        k, low, high = _grid(data, path)
        raw = np.stack([np.asarray(data['Pk0_raw'], float), np.asarray(data['Pk2'], float),
                        np.asarray(data['Pk4'], float)])
        _require(raw.shape == (3, k.size) and np.isfinite(raw).all(), 'Multipoles must be finite', path)
        n_objects, shot, sub = _shot_noise_checks(data, raw[0], path, layout)
        return dict(k=k, k_low=low, k_high=high, pk_raw=raw, pk0_sub=sub,
                    n_objects=np.int64(n_objects), shot_noise=np.float64(shot))


def _read_probabilities(path: Path, set_name: str, realization: int, layout: DataLayout):
    header, columns = read_fits_table(path, extname='ASTRA')
    _require(str(header.get('PARAM')) == set_name, f'ASTRA PARAM={header.get("PARAM")!r} != {set_name!r}', path)
    _require(int(header.get('REALIZ', -1)) == realization, 'ASTRA REALIZ does not match directory', path)
    _require(int(header.get('SNAPNUM', -1)) == layout.snapshot, 'ASTRA SNAPNUM mismatch', path)
    names = ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')
    _require(all(n in columns for n in names), 'Missing ASTRA probability columns', path)
    prob = np.stack([columns[n].astype(np.float64) for n in names], axis=1)
    n_iter = int(header['NITER'])
    _require(n_iter > 0 and len(prob) == int(header.get('SRCROWS', len(prob))) and len(prob) > 0,
             'ASTRA row count does not match SRCROWS', path)
    _require(np.isfinite(prob).all() and prob.min() >= -1e-6 and prob.max() <= 1 + 1e-6,
             'ASTRA probabilities outside [0, 1]', path)
    _require(np.allclose(prob.sum(axis=1), 1.0, atol=1e-4), 'ASTRA probabilities do not sum to 1', path)
    counts = np.rint(prob * n_iter)
    _require(np.abs(prob * n_iter - counts).max() < 1e-3, 'ASTRA probabilities are not multiples of 1/NITER', path)
    counts = counts.astype(np.int32)
    # argmax with ties resolved in (void, sheet, filament, knot) order, exactly as in the P(k) files.
    labels = np.argmax(counts, axis=1)
    ties = int(((counts == counts.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum())
    return counts, labels, ties, n_iter


def read_astra(path: Path, layout: DataLayout, set_name: str, realization: int, sample=None,
               expected_omega_m=None) -> dict:
    counts, labels, ties, n_iter = _read_probabilities(path, set_name, realization, layout)
    hist = np.stack([np.bincount(counts[:, j], minlength=n_iter + 1) for j in range(len(ENVIRONMENTS))])
    return dict(prob_hist=hist.astype(np.int32), class_counts=np.bincount(labels, minlength=4).astype(np.int64),
                ties=np.int64(ties), n_halos=np.int64(len(labels)), n_iter=np.int64(n_iter))


def _particle_mass(layout: DataLayout, omega_m: float, mnu: float, h: float) -> float:
    omega_nu = mnu / (93.14 * h * h)
    return RHO_CRIT * (omega_m - omega_nu) * (layout.box_size / layout.n_particles_side) ** 3


def read_hmf(path: Path, layout: DataLayout, set_name: str, realization: int, sample=None,
             expected_omega_m=None, astra_path: Path | None = None, cosmology: dict | None = None) -> dict:
    size = path.stat().st_size
    with open(path, 'rb') as handle:
        n_groups, tot_groups, _ = np.fromfile(handle, dtype=np.int32, count=3)
        np.fromfile(handle, dtype=np.int64, count=1)
        n_task = int(np.fromfile(handle, dtype=np.int32, count=1)[0])
    _require(n_task == 1 and n_groups == tot_groups and n_groups > 0, 'Expected a single-file FoF catalog', path)
    _require(size == 24 + 84 * int(n_groups), 'FoF group_tab size does not match its header', path)
    n = int(n_groups)
    lengths = np.fromfile(path, dtype=np.int32, count=n, offset=24)
    mass = np.fromfile(path, dtype=np.float32, count=n, offset=24 + 8 * n).astype(np.float64)
    _require(lengths.min() >= layout.min_halo_particles and np.isfinite(mass).all() and mass.min() > 0,
             'Invalid FoF halo lengths or masses', path)
    m_part = mass / lengths
    # GroupMass is a float32 sum: the largest halos carry ~1e-4 rounding error.
    _require(np.allclose(m_part, np.median(m_part), rtol=1e-3), 'FoF particle mass is not constant', path)
    mass = lengths * np.median(m_part)  # exact: one particle species
    if cosmology is not None:
        expected = _particle_mass(layout, cosmology['Omega_m'], cosmology.get('Mnu', 0.0), cosmology['h'])
        _require(np.isclose(np.median(m_part), expected, rtol=2e-3),
                 f'FoF particle mass {np.median(m_part):.4f}e10 inconsistent with Omega_m (expected {expected:.4f}e10)',
                 path)
    counts, labels, _, _ = _read_probabilities(astra_path, set_name, realization, layout)
    _require(len(labels) == n, f'ASTRA has {len(labels)} rows but the FoF catalog has {n} halos', path)
    log_mass = np.log10(mass) + 10.0
    _require(log_mass.max() < HMF_FINE_EDGES[-1] and log_mass.min() >= HMF_FINE_EDGES[0],
             'Halo masses fall outside the cached mass grid', path)
    hist = [np.histogram(log_mass, HMF_FINE_EDGES)[0]]
    hist += [np.histogram(log_mass[labels == j], HMF_FINE_EDGES)[0] for j in range(len(ENVIRONMENTS))]
    return dict(mass_hist=np.stack(hist).astype(np.int32), min_log_mass=np.float64(log_mass.min()),
                particle_mass=np.float64(np.median(m_part) * 1e10), n_halos=np.int64(n),
                class_counts=np.bincount(labels, minlength=4).astype(np.int64))


READERS = {'pk': read_pk, 'pk_rsd': read_pk_rsd, 'astra': read_astra, 'hmf': read_hmf}
SAMPLES = {'pk': HALO_SAMPLES + ('random_void',), 'pk_rsd': HALO_SAMPLES, 'astra': (None,), 'hmf': (None,)}
SHARED_KEYS = {'k', 'k_low', 'k_high', 'n_iter'}


def product_paths(layout: DataLayout, product: str, set_name: str, realization: int, sample) -> list[Path]:
    if product == 'hmf':
        return [layout.path('halos', set_name, realization), layout.path('astra', set_name, realization)]
    return [layout.path(product, set_name, realization, sample)]


def read_product(layout: DataLayout, product: str, set_name: str, realization: int, sample,
                 params: np.ndarray | None) -> dict:
    if set_name == LATIN_HYPERCUBE:
        cosmology = dict(zip(('Omega_m', 'Omega_b', 'h', 'n_s', 'sigma8'), params[realization])) | {'Mnu': 0.0}
    else:
        cosmology = set_cosmology(set_name)
    paths = product_paths(layout, product, set_name, realization, sample)
    if product == 'hmf':
        return read_hmf(paths[0], layout, set_name, realization, astra_path=paths[1], cosmology=cosmology)
    return READERS[product](paths[0], layout, set_name, realization, sample, cosmology['Omega_m'])
