'''Write a small synthetic Quijote tree with the same file formats as the real data.'''
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from src.config import DERIVATIVE_SETS, MNU_SETS, DataLayout, set_cosmology

ENVS = ('void', 'sheet', 'filament', 'knot')
N_BINS = 40
KF = 2 * np.pi / 1000.0


def _card(key, value):
    if isinstance(value, bool):
        text = 'T' if value else 'F'
        return f'{key:<8}= {text:>20}'.ljust(80)
    if isinstance(value, str):
        return f"{key:<8}= '{value:<8}'".ljust(80)
    return f'{key:<8}= {value!r:>20}'.ljust(80)


def _header(cards):
    text = ''.join(cards) + 'END'.ljust(80)
    text += ' ' * ((-len(text)) % 2880)
    return text.encode('ascii')


def write_astra(path: Path, probs: np.ndarray, set_name: str, realization: int, niter: int = 100):
    primary = _header([_card('SIMPLE', True), _card('BITPIX', 8), _card('NAXIS', 0), _card('EXTEND', True)])
    data = np.asarray(probs, dtype='>f4')
    cards = [_card('XTENSION', 'BINTABLE'), _card('BITPIX', 8), _card('NAXIS', 2), _card('NAXIS1', 16),
             _card('NAXIS2', len(data)), _card('PCOUNT', 0), _card('GCOUNT', 1), _card('TFIELDS', 4)]
    for i, name in enumerate(('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT'), 1):
        cards += [_card(f'TTYPE{i}', name), _card(f'TFORM{i}', 'E')]
    cards += [_card('EXTNAME', 'ASTRA'), _card('PARAM', set_name), _card('REALIZ', realization),
              _card('SNAPNUM', 3), _card('NITER', niter), _card('SRCROWS', len(data))]
    raw = data.tobytes()
    raw += b'\0' * ((-len(raw)) % 2880)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb') as handle:
        handle.write(primary + _header(cards) + raw)


def write_halos(path: Path, lengths: np.ndarray, particle_mass: float):
    n = len(lengths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as handle:
        np.array([n, n, int(lengths.sum())], dtype=np.int32).tofile(handle)
        np.array([lengths.sum()], dtype=np.int64).tofile(handle)
        np.array([1], dtype=np.int32).tofile(handle)
        lengths.astype(np.int32).tofile(handle)
        np.zeros(n, np.int32).tofile(handle)
        (lengths * particle_mass / 1e10).astype(np.float32).tofile(handle)
        np.zeros(3 * n, np.float32).tofile(handle)
        np.zeros(3 * n, np.float32).tofile(handle)
        np.zeros(6 * n, np.int32).tofile(handle)
        np.zeros(6 * n, np.float32).tofile(handle)


def _pk_file(path, set_name, realization, sample, pk_raw, n_objects, omega_m, lh: bool):
    centers = (np.arange(N_BINS) + 0.5) * 2 * KF
    shot = 1e9 / n_objects
    common = dict(k=centers * 1.01, k_center=centers, k_bin_low=centers - KF, k_bin_high=centers + KF,
                  Pk0_raw=pk_raw, Pk0_shot_subtracted=pk_raw - shot, Nmodes=np.ones(N_BINS),
                  shot_noise=np.float64(shot), n_objects=np.int64(n_objects), sample=np.str_(sample),
                  realization=np.int64(realization), snapshot=np.int64(3), box_size=np.float64(1000.0),
                  grid=np.int64(512), mas=np.str_('CIC'), space=np.str_('real'))
    if lh:  # PowerSpectrum style: Pk0 is raw, metadata has omega_m and redshift
        common.update(Pk0=pk_raw, parameter=np.str_(set_name), omega_m=np.float64(omega_m),
                      redshift=np.float64(0.5))
    else:   # PowerSpectrumReal style: Pk0 is shot-noise subtracted
        common.update(Pk0=pk_raw - shot, cosmology=np.str_(set_name))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **common)


def realization(layout: DataLayout, set_name: str, r: int, theta: dict, rng):
    om, s8, ns, h, ob = theta['Omega_m'], theta['sigma8'], theta['n_s'], theta['h'], theta['Omega_b']
    m_part = 27.7536627e10 * (om - theta.get('Mnu', 0) / (93.14 * h * h)) * (1000 / 512) ** 3
    n_halos = int(rng.poisson(1500 * (1 + 3 * (s8 - 0.8) - 1.5 * (om - 0.3))))
    lengths = (20 + rng.pareto(1.6, n_halos) * 15).astype(np.int64).clip(20, 400000)
    # ASTRA probability counts (multiples of 1/100), environment bias with sigma8.
    base = np.array([0.2, 0.4, 0.3, 0.1]) + np.array([-0.05, 0.0, 0.03, 0.02]) * (s8 - 0.8) / 0.2
    drawn = rng.choice(4, size=n_halos, p=base / base.sum())
    weights = 0.125 + 0.5 * np.eye(4)[drawn]
    counts = np.stack([rng.multinomial(100, w) for w in weights])
    labels = np.argmax(counts, axis=1)
    write_astra(layout.path('astra', set_name, r), counts / 100.0, set_name, r)
    write_halos(layout.path('halos', set_name, r), lengths, m_part)
    k = (np.arange(N_BINS) + 0.5) * 2 * KF
    amplitude = 2e4 * (s8 / 0.8) ** 2 * (om / 0.3) ** 0.5 * (k / 0.1) ** (ns - 2.0) * (1 + 0.3 * (h - 0.7) * k) \
        * (1 - 2 * (ob - 0.05) * np.sin(10 * k))
    lh = set_name == 'latin_hypercube'
    env_counts = np.bincount(labels, minlength=4)
    for sample, n_obj, bias in [('all', n_halos, 1.0)] + [(e, env_counts[i], 0.6 + 0.3 * i) for i, e in enumerate(ENVS)]:
        noise = np.exp(rng.normal(0, 0.03, N_BINS))
        pk = bias ** 2 * amplitude * noise + 1e9 / n_obj
        name = f'halo_{sample}'
        _pk_file(layout.path('pk', set_name, r, sample), set_name, r, name, pk, n_obj, om, lh)
    _pk_file(layout.path('pk', set_name, r, 'random_void'), set_name, r, 'random_void',
             0.5 * amplitude * np.exp(rng.normal(0, 0.03, N_BINS)) + 1e9 / (40 * env_counts[0] + 1),
             40 * env_counts[0] + 1, om, lh)


def build(root: Path, n_lh=60, n_fid=60, n_deriv=12, seed=0) -> DataLayout:
    rng = np.random.default_rng(seed)
    low, high = np.array([0.1, 0.03, 0.5, 0.8, 0.6]), np.array([0.5, 0.07, 0.9, 1.2, 1.0])
    params = low + (high - low) * rng.uniform(size=(n_lh, 5))
    params_file = root / 'params.txt'
    root.mkdir(parents=True, exist_ok=True)
    np.savetxt(params_file, params, fmt='%.5f')
    params = np.loadtxt(params_file)
    layout = DataLayout(root=root, params_file=params_file,
                        expected_realizations=(('latin_hypercube', n_lh), ('fiducial', n_fid)),
                        default_fixed_realizations=n_deriv)
    names = ('Omega_m', 'Omega_b', 'h', 'n_s', 'sigma8')
    for r in range(n_lh):
        realization(layout, 'latin_hypercube', r, dict(zip(names, params[r])), rng)
    sets = ['fiducial'] + [s for p, m, *_ in DERIVATIVE_SETS.values() for s in (p, m)] + [n for n, _ in MNU_SETS]
    for set_name in sets:
        for r in range(n_fid if set_name == 'fiducial' else n_deriv):
            # Matched seeds across fixed sets, as in Quijote.
            realization(layout, set_name, r, set_cosmology(set_name), np.random.default_rng(10_000 + r))
    return layout
