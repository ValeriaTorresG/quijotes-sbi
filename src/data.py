from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterable, Mapping, Protocol

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import ENVIRONMENT_FILES, PARAMETER_NAMES, SpectrumConfig


@dataclass
class PowerSpectrum:
    k: np.ndarray
    pk: np.ndarray
    bin_low: np.ndarray | None = None
    bin_high: np.ndarray | None = None


class SpectrumReader(Protocol):
    '''Implement read(path) to plug another spectrum format into the loader.'''
    def read(self, path: Path) -> PowerSpectrum: ...


class NPZSpectrumReader:
    '''Read the halo monopole format used by sbi_env and sbi_all.'''
    def __init__(self, k_key: str = 'k_center', pk_key: str = 'Pk0'):
        self.k_key, self.pk_key = k_key, pk_key

    def read(self, path: Path) -> PowerSpectrum:
        with np.load(path, allow_pickle=False) as data:
            return PowerSpectrum(k=np.asarray(data[self.k_key], dtype=float).reshape(-1),
                                 pk=np.asarray(data[self.pk_key], dtype=float).reshape(-1),
                                 bin_low=np.asarray(data['k_bin_low'], dtype=float).reshape(-1) if 'k_bin_low' in data else None,
                                 bin_high=np.asarray(data['k_bin_high'], dtype=float).reshape(-1) if 'k_bin_high' in data else None)


class TextSpectrumReader:
    '''Read matter Pk text files with columns k and Pk (sbi_init.ipynb).'''
    def __init__(self, k_column: int = 0, pk_column: int = 1, skiprows: int = 0):
        self.k_column, self.pk_column, self.skiprows = k_column, pk_column, skiprows

    def read(self, path: Path) -> PowerSpectrum:
        values = np.loadtxt(path, ndmin=2, skiprows=self.skiprows)
        return PowerSpectrum(k=values[:, self.k_column], pk=values[:, self.pk_column])


@dataclass
class SpectrumDataset:
    '''Features ordered by envs; theta row is params[cube_id], never file order.'''
    x: torch.Tensor
    theta: torch.Tensor
    cube_ids: np.ndarray
    envs: tuple[str, ...]
    k: dict[str, np.ndarray]
    feature_slices: dict[str, slice]
    config: SpectrumConfig
    parameter_names: tuple[str, ...] = PARAMETER_NAMES

    def __post_init__(self):
        if self.x.ndim != 2 or self.theta.ndim != 2:
            raise ValueError('x and theta must be two-dimensional.')

        if len(self.x) != len(self.theta) or len(self.x) != len(self.cube_ids) or not len(self.x):
            raise ValueError('x, theta and cube_ids must have matching, nonempty rows.')

        if self.theta.shape[1] != len(self.parameter_names):
            raise ValueError('theta columns must match parameter_names.')

        if not torch.isfinite(self.x).all() or not torch.isfinite(self.theta).all():
            raise ValueError('x and theta must be finite after float32 conversion.')

        if len(np.unique(self.cube_ids)) != len(self.cube_ids):
            raise ValueError('Duplicate cube IDs would leak simulations across splits.')

        start = 0
        for env in self.envs:
            stop = start + len(self.k[env])
            if self.feature_slices[env] != slice(start, stop):
                raise ValueError('Feature slices must follow envs and match their k grids.')
            start = stop

        if not self.envs or start != self.x.shape[1]:
            raise ValueError('Environment features must cover all x columns.')


    def select(self, envs: str | Iterable[str]) -> SpectrumDataset:
        '''Reuse loaded features for an individual environment or a combination.'''
        selected = _environment_names(envs)
        unknown = set(selected) - set(self.envs)
        if unknown:
            raise ValueError(f'Environments not loaded: {sorted(unknown)}')

        blocks, slices, start = [], {}, 0
        for env in selected:
            block = self.x[:, self.feature_slices[env]]
            blocks.append(block)
            slices[env] = slice(start, start + block.shape[1])
            start += block.shape[1]
        return SpectrumDataset(x=torch.cat(blocks, dim=1), theta=self.theta,
                               cube_ids=self.cube_ids.copy(), envs=selected,
                               k={env: self.k[env].copy() for env in selected}, feature_slices=slices,
                               config=self.config, parameter_names=self.parameter_names)

    def split(self, n_train: int, seed: int = 42) -> DatasetSplit:
        if isinstance(n_train, bool) or not isinstance(n_train, Integral) or not 1 <= n_train < len(self.x):
            raise ValueError(f'n_train must be an integer in [1, {len(self.x) - 1}].')
        rows = np.argsort(self.cube_ids)[np.random.default_rng(seed).permutation(len(self.x))]
        return DatasetSplit(self, rows[:n_train], rows[n_train:])


@dataclass
class DatasetSplit:
    data: SpectrumDataset
    train_rows: np.ndarray
    test_rows: np.ndarray

    @property
    def x_train(self):
        return self.data.x[self.train_rows]

    @property
    def theta_train(self):
        return self.data.theta[self.train_rows]

    @property
    def x_test(self):
        return self.data.x[self.test_rows]

    @property
    def theta_test(self):
        return self.data.theta[self.test_rows]

    @property
    def train_cube_ids(self):
        return self.data.cube_ids[self.train_rows]

    @property
    def test_cube_ids(self):
        return self.data.cube_ids[self.test_rows]


def _environment_names(envs: str | Iterable[str]) -> tuple[str, ...]:
    names = (envs,) if isinstance(envs, str) else tuple(envs)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError('Provide at least one nonempty environment name.')

    if len(set(names)) != len(names):
        raise ValueError('Environment names cannot repeat.')
    return names


class QuijoteDataLoader:
    '''Load base/<cube_id>/<env_filename> and align spectra to parameter rows.'''
    def __init__(self, base: str | Path, params_file: str | Path,
                 env_files: Mapping[str, str] | None = None,
                 config: SpectrumConfig | None = None,
                 reader: SpectrumReader | None = None,
                 parameter_names: Iterable[str] = PARAMETER_NAMES,
                 show_progress: bool = True):
        self.base, self.params_file = Path(base), Path(params_file)
        self.env_files = dict(ENVIRONMENT_FILES if env_files is None else env_files)
        self.config = config or SpectrumConfig()
        self.reader = reader or NPZSpectrumReader()
        self.parameter_names = tuple(parameter_names)
        self.show_progress = show_progress

    def load(self, envs: str | Iterable[str], cube_ids: Iterable[int]) -> SpectrumDataset:
        envs = _environment_names(envs)
        for env in envs:
            if env not in self.env_files:
                raise ValueError(f'Unknown environment {env!r}; available: {list(self.env_files)}')

        ids = list(cube_ids)
        if not ids or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or i < 0 for i in ids):
            raise ValueError('cube_ids must contain nonnegative integers.')

        if len(set(ids)) != len(ids):
            raise ValueError('cube_ids cannot repeat.')

        params = np.loadtxt(self.params_file, ndmin=2)
        if params.shape[1] != len(self.parameter_names):
            raise ValueError(f'Parameter file must have {len(self.parameter_names)} columns.')

        if max(ids) >= len(params):
            raise ValueError(f'cube_id {max(ids)} has no parameter row (only {len(params)} rows).')
        ids = np.asarray(ids, dtype=np.int64)
        theta = params[ids]

        if not np.isfinite(theta).all():
            raise ValueError('Selected cosmological parameters contain nonfinite values.')

        blocks, k_by_env, slices, start = [], {}, {}, 0
        for env in envs:
            reference = None
            rows = []
            for cube_id in tqdm(ids, desc=f'read {env}', disable=not self.show_progress):
                path = self.base / str(cube_id) / self.env_files[env]
                spectrum = self.reader.read(path)
                self._validate_spectrum(spectrum, path)

                if reference is None:
                    reference = spectrum
                    mask = (spectrum.k >= self.config.kmin) & (spectrum.k <= self.config.kmax)
                    if not mask.any():
                        raise ValueError(f'No bins in requested k range for {path}.')
                else:
                    self._validate_grid(reference, spectrum, path)
                pk = spectrum.pk[mask]

                if self.config.transform == 'log10':
                    if np.any(pk <= 0):
                        raise ValueError(f'log10 requires positive Pk in selected bins: {path}')
                    pk = np.log10(pk)
                rows.append(pk)
            block = np.stack(rows)
            blocks.append(block)
            k_by_env[env] = reference.k[mask].copy()
            slices[env] = slice(start, start + block.shape[1])
            start += block.shape[1]
        return SpectrumDataset(x=torch.as_tensor(np.concatenate(blocks, axis=1), dtype=torch.float32),
                               theta=torch.as_tensor(theta, dtype=torch.float32), cube_ids=ids, envs=envs,
                               k=k_by_env, feature_slices=slices, config=self.config,
                               parameter_names=self.parameter_names)

    @staticmethod
    def _validate_spectrum(spectrum: PowerSpectrum, path: Path):
        if spectrum.k.ndim != 1 or spectrum.pk.shape != spectrum.k.shape or not spectrum.k.size:
            raise ValueError(f'k and Pk must be matching nonempty vectors: {path}')

        if not np.isfinite(spectrum.k).all() or not np.isfinite(spectrum.pk).all():
            raise ValueError(f'Nonfinite k or Pk: {path}')

        if np.any(spectrum.k < 0) or np.any(np.diff(spectrum.k) <= 0):
            raise ValueError(f'k must be nonnegative and strictly increasing: {path}')

        if (spectrum.bin_low is None) != (spectrum.bin_high is None):
            raise ValueError(f'Provide both k_bin_low and k_bin_high, or neither: {path}')

        if spectrum.bin_low is not None:
            for edge in (spectrum.bin_low, spectrum.bin_high):
                if edge.shape != spectrum.k.shape or not np.isfinite(edge).all():
                    raise ValueError(f'Invalid bin edges: {path}')
            if np.any(spectrum.bin_low >= spectrum.bin_high):
                raise ValueError(f'Bin lower edges must lie below upper edges: {path}')
            if np.any(spectrum.k < spectrum.bin_low) or np.any(spectrum.k > spectrum.bin_high):
                raise ValueError(f'k centers must lie within bin edges: {path}')


    def _validate_grid(self, reference: PowerSpectrum, spectrum: PowerSpectrum, path: Path):
        for name in ('k', 'bin_low', 'bin_high'):
            ref, current = getattr(reference, name), getattr(spectrum, name)
            if ref is None and current is None:
                continue

            if (ref is None or current is None or ref.shape != current.shape
                or not np.allclose(ref, current, rtol=self.config.grid_rtol, atol=self.config.grid_atol)):
                raise ValueError(f'Inconsistent {name} grid between cubes: {path}')