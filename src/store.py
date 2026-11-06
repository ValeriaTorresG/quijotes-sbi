'''Parallel, validated and cached access to raw products of every Quijote set.

A cache file holds one (set, product, sample) table: stacked per-realization
arrays plus the realization IDs and a fingerprint (size and mtime of every
source file). The cache is rebuilt automatically when a source file changes,
when realizations appear or disappear, or when the reader version changes.
Caches store raw quantities only (full k grid, fine mass/probability grids),
so changing k cuts, transforms or bins never needs a rebuild.
'''
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .config import ALL_SETS, LATIN_HYPERCUBE, DataLayout
from .products import PRODUCTS, READER_VERSION, SAMPLES, SHARED_KEYS, DataValidationError, \
    product_paths, read_product


@dataclass
class ProductTable:
    '''Stacked raw arrays of one product for the realizations in ``ids``.'''
    set_name: str
    product: str
    sample: str | None
    ids: np.ndarray
    arrays: dict[str, np.ndarray]
    shared: dict[str, np.ndarray] = field(default_factory=dict)

    def take(self, ids) -> 'ProductTable':
        ids = np.asarray(ids, dtype=np.int64)
        position = {int(r): i for i, r in enumerate(self.ids)}
        missing = [int(r) for r in ids if int(r) not in position]
        if missing:
            raise KeyError(f'{self.set_name}/{self.product}/{self.sample}: realizations not available: '
                           f'{missing[:10]}{"..." if len(missing) > 10 else ""}')
        rows = np.array([position[int(r)] for r in ids], dtype=np.int64)
        return ProductTable(self.set_name, self.product, self.sample, ids.copy(),
                            {k: v[rows] for k, v in self.arrays.items()}, self.shared)


@dataclass
class Completeness:
    set_name: str
    product: str
    sample: str | None
    expected: int
    available: np.ndarray
    missing: np.ndarray

    @property
    def complete(self) -> bool:
        return len(self.missing) == 0


class ProductStore:
    def __init__(self, layout: DataLayout | None = None, cache_dir: str | Path | None = None,
                 workers: int = 16, show_progress: bool = True, verify_cache: bool = True):
        self.layout = layout or DataLayout()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[1] / 'cache'
        self.workers = max(1, int(workers))
        self.show_progress = show_progress
        self.verify_cache = verify_cache
        self._params = None
        self._memory: dict[tuple, ProductTable] = {}

    @property
    def params(self) -> np.ndarray:
        if self._params is None:
            params = np.loadtxt(self.layout.params_file, ndmin=2)
            if params.shape[1] != 5 or not np.isfinite(params).all():
                raise DataValidationError(f'Parameter file must have 5 finite columns: {self.layout.params_file}')
            self._params = params
        return self._params

    # ------------------------------------------------------------------ inventory
    def completeness(self, set_name: str, product: str, sample=None, n_expected: int | None = None) -> Completeness:
        _check_request(set_name, product, sample)
        n = n_expected or self.layout.n_expected(set_name)
        if set_name == LATIN_HYPERCUBE:
            n = min(n, len(self.params))
        ids = np.arange(n)
        with ThreadPoolExecutor(self.workers) as pool:
            present = list(pool.map(lambda r: all(p.is_file() for p in
                                                  product_paths(self.layout, product, set_name, int(r), sample)), ids))
        present = np.asarray(present, dtype=bool)
        return Completeness(set_name, product, sample, n, ids[present], ids[~present])

    # ------------------------------------------------------------------ loading
    def get(self, set_name: str, product: str, sample=None, ids=None, allow_missing: bool = False) -> ProductTable:
        '''Return validated raw arrays; ``ids=None`` means every expected realization.'''
        _check_request(set_name, product, sample)
        key = (set_name, product, sample)
        table = self._memory.get(key)
        if table is None:
            table = self._load_or_build(set_name, product, sample, allow_missing)
            self._memory[key] = table
        if ids is None:
            return table
        return table.take(ids)

    def _cache_path(self, set_name, product, sample) -> Path:
        tag = hashlib.sha1(json.dumps([str(self.layout.root), getattr(self.layout, 'pk_real_lh'),
                                       self.layout.pk_real_fixed, self.layout.pk_rsd, self.layout.astra,
                                       self.layout.halos, str(self.layout.params_file)]).encode()).hexdigest()[:10]
        return self.cache_dir / tag / set_name / f'{product}__{sample or "all"}.npz'

    def _fingerprint(self, set_name, product, sample, ids) -> np.ndarray:
        def stat(r):
            out = []
            for p in product_paths(self.layout, product, set_name, int(r), sample):
                s = os.stat(p)
                out += [s.st_size, s.st_mtime_ns]
            return out
        with ThreadPoolExecutor(self.workers) as pool:
            return np.asarray(list(pool.map(stat, ids)), dtype=np.int64)

    def _load_or_build(self, set_name, product, sample, allow_missing) -> ProductTable:
        status = self.completeness(set_name, product, sample)
        if not status.complete and not allow_missing:
            raise DataValidationError(
                f'{set_name}/{product}/{sample or ""}: {len(status.missing)} of {status.expected} realizations are '
                f'missing (first: {status.missing[:10].tolist()}). Use allow_missing=True to keep the '
                f'{len(status.available)} complete ones.')
        if not len(status.available):
            raise DataValidationError(f'No files for {set_name}/{product}/{sample or ""}.')
        ids = status.available
        path = self._cache_path(set_name, product, sample)
        fingerprint = self._fingerprint(set_name, product, sample, ids) if self.verify_cache else None
        if path.is_file():
            with np.load(path, allow_pickle=False) as cached:
                ok = (int(cached['reader_version']) == READER_VERSION and np.array_equal(cached['ids'], ids)
                      and (fingerprint is None or np.array_equal(cached['fingerprint'], fingerprint)))
                if ok:
                    arrays = {k[2:]: cached[k] for k in cached.files if k.startswith('a_')}
                    shared = {k[2:]: cached[k] for k in cached.files if k.startswith('s_')}
                    return ProductTable(set_name, product, sample, ids, arrays, shared)
        if fingerprint is None:
            fingerprint = self._fingerprint(set_name, product, sample, ids)
        table = self._build(set_name, product, sample, ids)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.tmp{os.getpid()}.npz')
        np.savez(tmp, ids=ids, fingerprint=fingerprint, reader_version=np.int64(READER_VERSION),
                 **{f'a_{k}': v for k, v in table.arrays.items()}, **{f's_{k}': v for k, v in table.shared.items()})
        os.replace(tmp, path)
        return table

    def _build(self, set_name, product, sample, ids) -> ProductTable:
        params = self.params if set_name == LATIN_HYPERCUBE else None

        def read(r):
            return read_product(self.layout, product, set_name, int(r), sample, params)

        with ThreadPoolExecutor(self.workers) as pool:
            records = list(tqdm(pool.map(read, ids), total=len(ids), disable=not self.show_progress,
                                desc=f'read {set_name}/{product}{"/" + sample if sample else ""}'))
        shared = {}
        for key in SHARED_KEYS & set(records[0]):
            reference = np.asarray(records[0][key])
            for r, record in zip(ids, records):
                if not np.allclose(record[key], reference, rtol=1e-6, atol=1e-10):
                    paths = product_paths(self.layout, product, set_name, int(r), sample)
                    raise DataValidationError(f'{key} differs from realization {ids[0]}: {paths[0]}')
            shared[key] = reference
        arrays = {key: np.stack([np.asarray(rec[key]) for rec in records])
                  for key in records[0] if key not in shared}
        return ProductTable(set_name, product, sample, np.asarray(ids, dtype=np.int64), arrays, shared)

    def inventory(self, sets=ALL_SETS, products=PRODUCTS):
        '''DataFrame of availability for every (set, product, sample).'''
        import pandas as pd
        rows = []
        for set_name in sets:
            for product in products:
                for sample in SAMPLES[product]:
                    status = self.completeness(set_name, product, sample)
                    rows.append(dict(set=set_name, product=product, sample=sample or '', expected=status.expected,
                                     available=len(status.available), missing=len(status.missing),
                                     first_missing=status.missing[:5].tolist()))
        return pd.DataFrame(rows)


def _check_request(set_name, product, sample):
    if set_name not in ALL_SETS:
        raise ValueError(f'Unknown set {set_name!r}; known: {ALL_SETS}')
    if product not in PRODUCTS:
        raise ValueError(f'Unknown product {product!r}; known: {PRODUCTS}')
    if sample not in SAMPLES[product]:
        raise ValueError(f'Product {product!r} does not take sample {sample!r}; allowed: {SAMPLES[product]}')
