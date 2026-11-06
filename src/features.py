'''Summary-statistic blocks and their assembly into datasets.

A block is ``<statistic>:<sample>``:

=========  ====================================  ===========================================
statistic  samples                               features
=========  ====================================  ===========================================
pk         all, void, sheet, filament, knot,     T[P0(k)], real space, kmin <= k <= kmax
           random_void                           (T = asinh(P/s) of shot-noise-subtracted P by default)
pkrsd      all, void, sheet, filament, knot      T[P0(k)] and P_l(k)/P0_raw(k) (l in rsd_multipoles)
nbar       same as pk                            log10 number of objects (abundance)
hmf        all, void, sheet, filament, knot      log10(1 + N) halos per log10-mass bin
astra      void, sheet, filament, knot           log10(1 + N) halos per ASTRA-probability bin
=========  ====================================  ===========================================

Shorthands: a bare sample (``void``) means ``pk:void`` (backwards compatible),
and ``envs`` expands to the four environments (``pk:envs`` -> four blocks).
The same transformation is applied to the latin hypercube and to the fixed
fiducial/derivative sets, so NPE, Fisher and the fiducial checks see
identical features.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .config import ENVIRONMENTS, HALO_SAMPLES, LATIN_HYPERCUBE, PARAMETER_NAMES, PK_SAMPLES, FeatureConfig, \
    set_cosmology
from .data import SpectrumDataset
from .products import HMF_FINE_EDGES, DataValidationError
from .store import ProductStore

STATISTICS = {'pk': PK_SAMPLES, 'pkrsd': HALO_SAMPLES, 'nbar': PK_SAMPLES, 'hmf': HALO_SAMPLES,
              'astra': ENVIRONMENTS}
_PRODUCT = {'pk': 'pk', 'pkrsd': 'pk_rsd', 'nbar': 'pk', 'hmf': 'hmf', 'astra': 'astra'}

PRESETS = {
    'environments': {s: f'pk:{s}' for s in PK_SAMPLES} | {
        'pk_envs': 'pk:envs', 'pk_envs_all': 'pk:envs,pk:all'},
    'full': {
        'pk_all': 'pk:all', 'pk_envs': 'pk:envs', 'pk_envs_all': 'pk:envs,pk:all',
        'pk_random_void': 'pk:random_void',
        'pk_envs_nbar': 'pk:envs,nbar:envs',
        'hmf_all': 'hmf:all', 'hmf_envs': 'hmf:envs', 'astra_envs': 'astra:envs',
        'pk_hmf_envs': 'pk:envs,nbar:envs,hmf:envs',
        'everything': 'pk:all,pk:envs,pk:random_void,nbar:all,nbar:envs,nbar:random_void,hmf:envs,astra:envs'},
}


@dataclass(frozen=True)
class Block:
    statistic: str
    sample: str

    @property
    def name(self) -> str:
        return f'{self.statistic}:{self.sample}'

    @property
    def product(self) -> str:
        return _PRODUCT[self.statistic]

    @property
    def product_sample(self):
        return None if self.statistic in ('hmf', 'astra') else self.sample


def parse_blocks(spec: str | Iterable[str]) -> tuple[Block, ...]:
    '''Parse ``'pk:envs,nbar:all'`` (or a list of tokens / Blocks) into distinct blocks.'''
    if isinstance(spec, str):
        tokens = spec.split(',')
    else:
        tokens = [t for item in spec for t in (item.name if isinstance(item, Block) else str(item)).split(',')]
    blocks: list[Block] = []
    for token in (t.strip() for t in tokens):
        if not token:
            continue
        statistic, _, sample = token.partition(':') if ':' in token else ('pk', '', token)
        if statistic not in STATISTICS:
            raise ValueError(f'Unknown statistic {statistic!r} in {token!r}; known: {sorted(STATISTICS)}')
        samples = ENVIRONMENTS if sample == 'envs' else (sample,)
        for s in samples:
            if s not in STATISTICS[statistic]:
                raise ValueError(f'{statistic} does not support sample {s!r}; allowed: {STATISTICS[statistic]}')
            block = Block(statistic, s)
            if block in blocks:
                raise ValueError(f'Block {block.name} appears twice in {spec!r}.')
            blocks.append(block)
    if not blocks:
        raise ValueError('Provide at least one summary-statistic block.')
    return tuple(blocks)


def parse_experiments(items: Iterable[str]) -> dict[str, tuple[Block, ...]]:
    '''``['void=pk:void', 'pk:envs,nbar:envs']`` -> {name: blocks}. Unnamed specs get a derived name.'''
    experiments = {}
    for item in items:
        name, sep, spec = item.partition('=')
        if not sep:
            name, spec = item.replace(':', '-').replace(',', '+'), item
        name = name.strip()
        if not name or name in experiments:
            raise ValueError(f'Experiment names must be nonempty and unique: {name!r}')
        experiments[name] = parse_blocks(spec)
    if not experiments:
        raise ValueError('Provide at least one experiment.')
    return experiments


def _rebin(fine_edges: np.ndarray, edges: np.ndarray, what: str) -> np.ndarray:
    index = np.searchsorted(fine_edges, edges)
    ok = (index < len(fine_edges)) & np.isclose(fine_edges[np.minimum(index, len(fine_edges) - 1)], edges, atol=1e-6)
    if not ok.all():
        raise ValueError(f'{what} edges must lie on the cached grid {fine_edges[0]}..{fine_edges[-1]} '
                         f'(step {fine_edges[1] - fine_edges[0]:g}); got {edges.tolist()}')
    return index


def block_features(block: Block, table, config: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    '''Return (features [n, d], abscissa [d]) for one block of one set.'''
    a, ids = table.arrays, table.ids
    spec = config.spectrum
    if block.statistic in ('pk', 'pkrsd'):
        k = table.shared['k']
        mask = (k >= spec.kmin) & (k <= spec.kmax)
        if not mask.any():
            raise ValueError(f'No k bins in [{spec.kmin}, {spec.kmax}] for {block.name}.')
        if block.statistic == 'pk':
            pk0 = (a['pk_raw'] if spec.shot_noise == 'raw' else a['pk_sub'])[:, mask]
        else:
            pk0 = (a['pk_raw'][:, 0] if spec.shot_noise == 'raw' else a['pk0_sub'])[:, mask]
        if spec.transform == 'log10':
            bad = ids[np.any(pk0 <= 0, axis=1)]
            if len(bad):
                raise DataValidationError(
                    f'{block.name}: log10 needs positive P0 in the selected bins; realizations {bad[:10].tolist()} '
                    f'have P0 <= 0 (use transform="asinh", shot_noise="raw" or a smaller kmax).')
            pk0 = np.log10(pk0)
        elif spec.transform == 'asinh':
            pk0 = np.arcsinh(pk0 / spec.asinh_scale)
        if block.statistic == 'pk':
            return pk0, k[mask]
        raw0 = a['pk_raw'][:, 0, mask]
        if np.any(raw0 <= 0):
            raise DataValidationError(f'{block.name}: raw P0 must be positive to form P_l / P0.')
        parts = [pk0 if ell == 0 else a['pk_raw'][:, ell // 2, mask] / raw0 for ell in config.rsd_multipoles]
        return np.concatenate(parts, axis=1), np.tile(k[mask], len(parts))
    if block.statistic == 'nbar':
        return np.log10(a['n_objects'].astype(float))[:, None], np.zeros(1)
    if block.statistic == 'astra':
        hist = a['prob_hist'][:, ENVIRONMENTS.index(block.sample)].astype(float)
        n_levels = hist.shape[1]
        bins = np.minimum(np.arange(n_levels) * config.astra_bins // n_levels, config.astra_bins - 1)
        counts = np.stack([hist[:, bins == b].sum(axis=1) for b in range(config.astra_bins)], axis=1)
        centers = (np.arange(config.astra_bins) + 0.5) / config.astra_bins
        return np.log10(1.0 + counts), centers
    if block.statistic == 'hmf':
        edges = np.asarray(config.hmf_log10_mass_edges)
        index = _rebin(HMF_FINE_EDGES, edges, 'hmf_log10_mass_edges')
        threshold = np.log10(table.arrays['particle_mass'] * 20)
        if not config.allow_incomplete_hmf and edges[0] < threshold.max() - 1e-9:
            worst = ids[np.argmax(threshold)]
            raise DataValidationError(
                f'hmf: the lowest mass edge 10^{edges[0]} is below the 20-particle FoF threshold '
                f'10^{threshold.max():.3f} of realization {worst} in {table.set_name}; raise the first edge or set '
                f'allow_incomplete_hmf=True.')
        hist = a['mass_hist'][:, HALO_SAMPLES.index(block.sample)].astype(float)
        cumulative = np.concatenate([np.zeros((len(hist), 1)), np.cumsum(hist, axis=1)], axis=1)
        counts = np.diff(cumulative[:, index], axis=1)
        return np.log10(1.0 + counts), 0.5 * (edges[1:] + edges[:-1])
    raise ValueError(block.statistic)


@dataclass
class FixedSetData:
    '''Features of a fixed-cosmology set, transformed exactly like the training data.'''
    set_name: str
    x: np.ndarray          # float64: finite differences need the precision
    ids: np.ndarray
    blocks: tuple[str, ...]
    cosmology: dict[str, float]
    block_sizes: tuple[int, ...] = ()

    @property
    def x_tensor(self) -> torch.Tensor:
        return torch.as_tensor(self.x, dtype=torch.float32)

    def theta(self, parameter_names=PARAMETER_NAMES) -> np.ndarray:
        return np.array([self.cosmology[p] for p in parameter_names], dtype=float)


class QuijoteSummaryLoader:
    '''Build feature matrices for any combination of blocks and any Quijote set.'''

    def __init__(self, store: ProductStore | None = None, config: FeatureConfig | None = None,
                 allow_missing: bool = False, parameter_names=PARAMETER_NAMES):
        self.store = store or ProductStore()
        self.config = config or FeatureConfig()
        self.allow_missing = allow_missing
        self.parameter_names = tuple(parameter_names)

    @staticmethod
    def block_names(spec) -> tuple[str, ...]:
        '''Canonical block names of a spec (``'pk:envs'`` -> four ``pk:<env>`` names).'''
        if _is_blocks(spec):
            return tuple(b.name for b in spec)
        return tuple(b.name for b in parse_blocks(spec))

    def _tables(self, set_name: str, blocks: tuple[Block, ...]):
        return {b: self.store.get(set_name, b.product, b.product_sample, allow_missing=self.allow_missing)
                for b in blocks}

    def available_ids(self, set_name: str, blocks) -> np.ndarray:
        blocks = parse_blocks(blocks) if not _is_blocks(blocks) else blocks
        ids = None
        for table in self._tables(set_name, blocks).values():
            ids = table.ids if ids is None else np.intersect1d(ids, table.ids)
        return ids

    def _features(self, set_name, blocks, ids):
        tables = {b: t.take(ids) for b, t in self._tables(set_name, blocks).items()}
        check_consistency(tables)
        parts, abscissa, slices, start = [], {}, {}, 0
        for block in blocks:
            values, axis = block_features(block, tables[block], self.config)
            if not np.isfinite(values).all():
                bad = ids[~np.isfinite(values).all(axis=1)]
                raise DataValidationError(f'{set_name}/{block.name}: non-finite features in {bad[:10].tolist()}')
            parts.append(values)
            abscissa[block.name] = axis
            slices[block.name] = slice(start, start + values.shape[1])
            start += values.shape[1]
        return np.concatenate(parts, axis=1), abscissa, slices

    def load(self, blocks, cube_ids=None) -> SpectrumDataset:
        '''Latin-hypercube dataset; theta rows are ``params[cube_id]``.'''
        blocks = parse_blocks(blocks) if not _is_blocks(blocks) else blocks
        available = self.available_ids(LATIN_HYPERCUBE, blocks)
        if cube_ids is None:
            ids = available
        else:
            ids = np.asarray(list(cube_ids), dtype=np.int64)
            if len(np.unique(ids)) != len(ids):
                raise ValueError('cube_ids cannot repeat.')
            missing = np.setdiff1d(ids, available)
            if len(missing):
                if not self.allow_missing:
                    raise DataValidationError(f'Requested cubes lack data for {[b.name for b in blocks]}: '
                                              f'{missing[:10].tolist()}')
                ids = ids[np.isin(ids, available)]
        x, abscissa, slices = self._features(LATIN_HYPERCUBE, blocks, ids)
        theta = self.store.params[ids]
        return SpectrumDataset(x=torch.as_tensor(x, dtype=torch.float32),
                               theta=torch.as_tensor(theta, dtype=torch.float32), cube_ids=ids.copy(),
                               envs=tuple(b.name for b in blocks), k=abscissa, feature_slices=slices,
                               config=self.config, parameter_names=self.parameter_names)

    def load_set(self, set_name: str, blocks, ids=None) -> FixedSetData:
        blocks = parse_blocks(blocks) if not _is_blocks(blocks) else blocks
        if set_name == LATIN_HYPERCUBE:
            raise ValueError('Use load() for the latin hypercube.')
        ids = self.available_ids(set_name, blocks) if ids is None else np.asarray(ids, dtype=np.int64)
        x, _, slices = self._features(set_name, blocks, ids)
        return FixedSetData(set_name, np.asarray(x, dtype=np.float64), ids.copy(),
                            tuple(b.name for b in blocks), set_cosmology(set_name),
                            tuple(s.stop - s.start for s in slices.values()))


def _is_blocks(blocks) -> bool:
    return isinstance(blocks, tuple) and all(isinstance(b, Block) for b in blocks)


def check_consistency(tables: dict) -> None:
    '''Cross-product checks on the same realizations (counts must agree exactly).'''
    by_key = {(b.statistic, b.sample): t for b, t in tables.items()}
    pk_counts = {b.sample: t.arrays['n_objects'] for b, t in tables.items() if b.product in ('pk', 'pk_rsd')}
    if all(s in pk_counts for s in HALO_SAMPLES):
        total = sum(pk_counts[s] for s in ENVIRONMENTS)
        bad = np.flatnonzero(total != pk_counts['all'])
        if len(bad):
            ids = next(iter(tables.values())).ids
            raise DataValidationError(f'Environment halo counts do not add up to the full catalog for '
                                      f'realizations {ids[bad][:10].tolist()}.')
    for (statistic, _), table in by_key.items():
        if statistic in ('astra', 'hmf'):
            class_counts = table.arrays['class_counts']
            for j, env in enumerate(ENVIRONMENTS):
                if env in pk_counts:
                    bad = np.flatnonzero(class_counts[:, j] != pk_counts[env])
                    if len(bad):
                        raise DataValidationError(
                            f'ASTRA {env} halo count differs from the P(k) file for realizations '
                            f'{table.ids[bad][:10].tolist()} ({table.set_name}).')
