import os
import sys
import tempfile
import time
import zlib
import unittest
from pathlib import Path

import numpy as np
import torch

from src import (DataValidationError, EvaluationConfig, FeatureConfig, FisherConfig, FisherForecaster,
                 FixedSetData, PriorConfig, ProductStore, QuijoteSummaryLoader, ScoreCompressor, SpectrumConfig,
                 parse_blocks, parse_experiments)
from src.evaluation import evaluate_samples
from src.features import PRESETS
from src.fits import read_fits_table
from src.products import read_product

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_quijote import build, write_astra  # noqa: E402

SMALL = FeatureConfig(spectrum=SpectrumConfig(kmin=0.01, kmax=0.15), astra_bins=5,
                      hmf_log10_mass_edges=(13.4, 13.8, 14.2, 17.0))


class SyntheticTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.layout = build(cls.root / 'data')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def store(self, **kwargs):
        return ProductStore(self.layout, self.root / 'cache', workers=4, show_progress=False, **kwargs)


class FitsAndReaders(SyntheticTree):
    def test_fits_roundtrip(self):
        probs = np.array([[0.1, 0.2, 0.3, 0.4], [1.0, 0.0, 0.0, 0.0]])
        path = self.root / 'x.fits.gz'
        write_astra(path, probs, 'fiducial', 3)
        header, columns = read_fits_table(path, 'ASTRA')
        self.assertEqual(header['PARAM'], 'fiducial')
        self.assertEqual(header['REALIZ'], 3)
        np.testing.assert_allclose(columns['PSHEET'], probs[:, 1], rtol=1e-6)

    def test_reader_rejects_misplaced_files(self):
        params = np.loadtxt(self.layout.params_file)
        record = read_product(self.layout, 'pk', 'latin_hypercube', 2, 'void', params)
        self.assertEqual(record['pk_raw'].shape, (40,))
        wrong = params.copy()
        wrong[2, 0] += 0.01  # Omega_m in the file no longer matches the parameter row
        with self.assertRaisesRegex(DataValidationError, 'Omega_m'):
            read_product(self.layout, 'pk', 'latin_hypercube', 2, 'void', wrong)
        # A file whose metadata says realization 5 in directory 4.
        source = self.layout.path('pk', 'fiducial', 5, 'knot')
        target_dir = self.root / 'bad'
        layout = type(self.layout)(root=target_dir, params_file=self.layout.params_file)
        target = layout.path('pk', 'fiducial', 4, 'knot')
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(DataValidationError, 'realization'):
            read_product(layout, 'pk', 'fiducial', 4, 'knot', None)

    def test_raw_pk_is_read_explicitly(self):
        '''Pk0 means raw in the LH files and shot-subtracted in the fixed sets; we must not depend on it.'''
        lh = read_product(self.layout, 'pk', 'latin_hypercube', 0, 'all', np.loadtxt(self.layout.params_file))
        fid = read_product(self.layout, 'pk', 'fiducial', 0, 'all', None)
        for record in (lh, fid):
            np.testing.assert_allclose(record['pk_raw'] - record['pk_sub'], record['shot_noise'])

    def test_hmf_checks_particle_mass_and_rows(self):
        params = np.loadtxt(self.layout.params_file)
        record = read_product(self.layout, 'hmf', 'latin_hypercube', 1, None, params)
        self.assertEqual(record['mass_hist'].shape[0], 5)
        self.assertEqual(record['mass_hist'][0].sum(), record['n_halos'])
        self.assertEqual(record['mass_hist'][1:].sum(), record['n_halos'])
        wrong = params.copy()
        wrong[1, 0] *= 1.2
        with self.assertRaises(DataValidationError):
            read_product(self.layout, 'hmf', 'latin_hypercube', 1, None, wrong)


class StoreAndFeatures(SyntheticTree):
    def test_cache_reuse_and_invalidation(self):
        store = self.store()
        table = store.get('fiducial', 'pk', 'void')
        self.assertEqual(len(table.ids), 60)
        cached = store._cache_path('fiducial', 'pk', 'void')
        self.assertTrue(cached.is_file())
        stamp = cached.stat().st_mtime_ns
        self.store().get('fiducial', 'pk', 'void')
        self.assertEqual(cached.stat().st_mtime_ns, stamp, 'unchanged sources must reuse the cache')
        source = self.layout.path('pk', 'fiducial', 7, 'void')
        time.sleep(0.01)
        os.utime(source)
        self.store().get('fiducial', 'pk', 'void')
        self.assertNotEqual(cached.stat().st_mtime_ns, stamp, 'a modified source must rebuild the cache')

    def test_missing_realizations(self):
        store = self.store()
        path = self.layout.path('pk', 'Om_p', 3, 'sheet')
        moved = path.with_suffix('.bak')
        path.rename(moved)
        try:
            with self.assertRaisesRegex(DataValidationError, 'missing'):
                store.get('Om_p', 'pk', 'sheet')
            table = self.store().get('Om_p', 'pk', 'sheet', allow_missing=True)
            self.assertNotIn(3, table.ids.tolist())
        finally:
            moved.rename(path)

    def test_blocks_and_presets(self):
        names = [b.name for b in parse_blocks('pk:envs,nbar:all')]
        self.assertEqual(names, ['pk:void', 'pk:sheet', 'pk:filament', 'pk:knot', 'nbar:all'])
        self.assertEqual([b.name for b in parse_blocks(['random_void', 'hmf:knot'])], ['pk:random_void', 'hmf:knot'])
        with self.assertRaisesRegex(ValueError, 'twice'):
            parse_blocks('pk:void,void')
        with self.assertRaises(ValueError):
            parse_blocks('astra:all')
        with self.assertRaises(ValueError):
            parse_blocks('foo:void')
        for preset in PRESETS.values():
            parse_experiments(f'{n}={s}' for n, s in preset.items())

    def test_dataset_assembly_and_consistency(self):
        loader = QuijoteSummaryLoader(self.store(), SMALL)
        data = loader.load('pk:envs,pk:all,nbar:envs,astra:void,hmf:all,hmf:knot')
        self.assertEqual(len(data.cube_ids), 60)
        np.testing.assert_allclose(data.theta.numpy(), np.loadtxt(self.layout.params_file)[data.cube_ids],
                                   rtol=1e-6)
        n_k = int(((np.arange(40) + 0.5) * 4 * np.pi / 1000 >= 0.01).sum() -
                  ((np.arange(40) + 0.5) * 4 * np.pi / 1000 > 0.15).sum())
        self.assertEqual(data.feature_slices['pk:void'].stop, n_k)
        self.assertEqual(data.x.shape[1], 5 * n_k + 4 + 5 + 3 + 3)
        sub = data.select(['nbar:void', 'pk:knot'])
        torch.testing.assert_close(sub.x[:, 0], data.x[:, data.feature_slices['nbar:void']][:, 0])
        fixed = loader.load_set('Om_p', 'pk:void,nbar:void')
        self.assertIsInstance(fixed, FixedSetData)
        self.assertEqual(fixed.cosmology['Omega_m'], 0.3275)
        self.assertEqual(fixed.x.dtype, np.float64)

    def test_hmf_threshold_guard(self):
        config = FeatureConfig(spectrum=SMALL.spectrum, hmf_log10_mass_edges=(12.0, 13.0, 17.0))
        loader = QuijoteSummaryLoader(self.store(), config)
        with self.assertRaisesRegex(DataValidationError, 'threshold'):
            loader.load('hmf:all')


class FakeLoader:
    '''Linear-Gaussian model: x = A theta + noise, with exact Fisher A^T C^-1 A.'''

    def __init__(self, n_fid=4000, n_der=400, seed=1, d=8, matched=True, derivative_noise=1.0):
        rng = np.random.default_rng(seed)
        self.A = rng.normal(size=(d, 5))
        self.C = np.diag(rng.uniform(0.5, 2.0, d))
        self.n_fid, self.n_der, self.d, self.matched = n_fid, n_der, d, matched
        self.derivative_noise = derivative_noise

    def load_set(self, set_name, blocks):
        from src.config import set_cosmology, PARAMETER_NAMES
        cosmology = set_cosmology(set_name)
        theta = np.array([cosmology[p] for p in PARAMETER_NAMES])
        n = self.n_fid if set_name == 'fiducial' else self.n_der
        seed = 5 if set_name != 'fiducial' else 6
        if not self.matched:  # independent seeds per set: noisy finite differences
            seed = zlib.crc32(set_name.encode()) % 10_000 + 100
        noise = np.random.default_rng(seed).multivariate_normal(np.zeros(self.d), self.C, size=n)
        if set_name != 'fiducial':
            noise *= self.derivative_noise
        return FixedSetData(set_name, self.A @ theta + noise, np.arange(n), ('pk:void',), cosmology, (self.d,))


class FisherTest(unittest.TestCase):
    def test_linear_gaussian_fisher(self):
        loader = FakeLoader()
        result = FisherForecaster(loader, FisherConfig()).compute('pk:void')
        exact = loader.A.T @ np.linalg.inv(loader.C) @ loader.A
        np.testing.assert_allclose(result.sigma, np.sqrt(np.diag(np.linalg.inv(exact))), rtol=0.06)
        np.testing.assert_allclose(result.derivatives, loader.A.T, atol=1e-8)  # matched seeds: exact
        compressor = ScoreCompressor.from_fisher(result)
        theta = np.array([0.3, 0.05, 0.7, 0.95, 0.8])
        fiducial = np.array([0.3175, 0.049, 0.6711, 0.9624, 0.834])
        # Data with the same noise offset as the estimated fiducial mean -> t recovers theta exactly.
        x = loader.A @ (theta - fiducial) + result.mean
        t = compressor(torch.tensor(x)[None]).numpy()[0]
        np.testing.assert_allclose(t, theta, atol=1e-5)
        np.testing.assert_allclose(result.sigma_compressed, result.sigma, rtol=0.05)  # exact derivatives
        table = FisherForecaster(loader, FisherConfig()).convergence('pk:void', fractions=(0.5, 1.0))
        self.assertTrue(np.allclose(table[table.fraction == 1.0].ratio_to_full, 1.0))


    def test_derivative_noise_brackets(self):
        '''Noisy derivatives: standard Fisher is optimistic, compressed is conservative, combined in between.'''
        loader = FakeLoader(n_fid=6000, n_der=100, d=60, matched=False, seed=3, derivative_noise=0.03)
        exact = np.sqrt(np.diag(np.linalg.inv(loader.A.T @ np.linalg.inv(loader.C) @ loader.A)))
        result = FisherForecaster(loader, FisherConfig()).compute('pk:void')
        j = 1  # Omega_b: smallest step, noisiest derivative
        ratio_std, ratio_comp, ratio_comb = (result.sigma[j] / exact[j], result.sigma_compressed[j] / exact[j],
                                             result.sigma_combined[j] / exact[j])
        self.assertLess(ratio_std, 0.9)
        self.assertGreater(ratio_comp, 1.1)
        self.assertLess(abs(np.log(ratio_comb)), min(abs(np.log(ratio_std)), abs(np.log(ratio_comp))))


class ExactGaussianPosterior:
    '''Conjugate posterior of theta ~ U-free Gaussian model, used to test the metrics.'''

    def __init__(self, sigma=0.05):
        self.sigma = sigma
        self.prior = PriorConfig(low=(-10.0, -10.0), high=(10.0, 10.0))
        self.parameter_names = ('a', 'b')

    def sample_batched(self, x, n, seed=None):
        g = torch.Generator().manual_seed(seed or 0)
        x = torch.as_tensor(x)
        return x[None] + self.sigma * torch.randn((n, *x.shape), generator=g)

    def log_prob(self, theta, x, leakage_samples=0, acceptance=None):
        z = (theta - torch.as_tensor(x)) / self.sigma
        return (-0.5 * z ** 2 - np.log(self.sigma * np.sqrt(2 * np.pi))).sum(-1)

    def acceptance(self, x, n):
        return torch.ones(1, len(x))


class EvaluationTest(unittest.TestCase):
    def test_metrics_detect_calibration(self):
        rng = np.random.default_rng(0)
        truth = rng.uniform(-1, 1, (600, 2))
        model = ExactGaussianPosterior()
        x = torch.as_tensor(truth + rng.normal(0, model.sigma, truth.shape), dtype=torch.float32)
        config = EvaluationConfig(n_samples=400, show_progress=False, batch_size=100, n_tarp_references=2)
        good = evaluate_samples(model, x, truth, np.arange(600), config)
        s = good.summary
        self.assertTrue(np.all(np.abs(s.coverage_68 - 0.68) < 0.06))
        self.assertTrue(np.all(np.abs(s.coverage_95 - 0.95) < 0.04))
        self.assertTrue(np.all(s.pit_ks_pvalue > 0.001))
        self.assertLess(good.scores['tarp_max_deviation'], 0.08)
        overconfident = ExactGaussianPosterior(sigma=0.02)
        bad = evaluate_samples(overconfident, x, truth, np.arange(600), config)
        self.assertTrue(np.all(bad.summary.coverage_68 < 0.45))
        self.assertTrue(np.all(bad.summary.pit_ks_pvalue < 1e-6))
        self.assertGreater(bad.scores['tarp_max_deviation'], 0.15)
        self.assertGreater(bad.scores['nll'], good.scores['nll'])


class CommandLine(SyntheticTree):
    def cli(self, *args):
        from src.__main__ import main
        common = ['--root', str(self.layout.root), '--params-file', str(self.layout.params_file),
                  '--cache-dir', str(self.root / 'cli_cache'), '--allow-missing', '--workers', '4']
        return main([args[0], *common, *args[1:]])

    def test_fisher_and_inventory(self):
        out = self.root / 'cli_fisher'
        with self.assertRaises(SystemExit):  # 4 features cannot constrain 5 parameters
            self.cli('fisher', '-e', 'n=nbar:envs', '--output', str(self.root / 'cli_bad'))
        self.assertEqual(self.cli('fisher', '-e', 'v=pk:void', 'vk=pk:void,pk:knot', 'n=nbar:envs,pk:all',
                                  '--kmax', '0.15', '--output', str(out), '--convergence'), 0)
        import pandas as pd
        table = pd.read_csv(out / 'fisher_comparison.csv')
        self.assertEqual(set(table.experiment), {'v', 'vk', 'n'})
        self.assertTrue(np.isfinite(table.sigma_fisher).all())
        self.assertTrue((out / 'fisher_convergence.csv').is_file())
        self.assertEqual(self.cli('inventory', '--sets', 'fiducial', '--products', 'pk',
                                  '--output', str(self.root / 'inv.csv')), 0)

    @unittest.skipUnless(os.environ.get('QUIJOTE_RUN_SBI_SMOKE'), 'set QUIJOTE_RUN_SBI_SMOKE=1 to train small flows')
    def test_train_overlapping_experiments_both_methods(self):
        out = self.root / 'cli_train'
        code = self.cli('train', '-e', 'v=pk:void', 'vk=pk:void,pk:knot', '--method', 'both', '--kmax', '0.15',
                        '--ensemble', '2', '--hidden-features', '8', '--num-transforms', '1', '--max-epochs', '5',
                        '--patience', '2', '--n-samples', '32', '--fiducial', '5', '--fisher', '--device', 'cpu',
                        '--n-train', '50', '--output', str(out))
        self.assertEqual(code, 0)
        import pandas as pd
        comparison = pd.read_csv(out / 'comparison.csv')
        self.assertEqual(set(comparison.method), {'npe', 'mn'})
        self.assertTrue((out / 'npe' / 'vk' / 'model.pt').is_file())
        self.assertTrue((out / 'mn' / 'v' / 'fiducial_summary.csv').is_file())


@unittest.skipUnless(os.environ.get('QUIJOTE_RUN_SBI_SMOKE'), 'set QUIJOTE_RUN_SBI_SMOKE=1 to train small flows')
class EndToEndSmoke(SyntheticTree):
    def test_npe_ensemble_with_moped_fiducial_and_reload(self):
        from src import SBIPipeline, SBITrainer, PosteriorEvaluator, TrainingConfig, TrainedPosterior
        torch.set_num_threads(1)
        loader = QuijoteSummaryLoader(self.store(), SMALL)
        forecaster = FisherForecaster(loader)
        trainer = SBITrainer(TrainingConfig(n_train=50, hidden_features=16, num_transforms=2, batch_size=16,
                                            max_epochs=15, patience=5, ensemble_size=2, compression='moped',
                                            show_train_summary=False))
        pipeline = SBIPipeline(loader, trainer, PosteriorEvaluator(EvaluationConfig(n_samples=64, show_progress=False,
                                                                                    leakage_samples=64)),
                               fisher=forecaster, n_fiducial=10)
        results = pipeline.run_many({'void': 'pk:void', 'combo': 'pk:void,nbar:envs'})
        result = results['combo']
        self.assertEqual(len(result.fit.model.members), 2)
        self.assertEqual(len(result.evaluation.truth), 10)
        self.assertEqual(result.fiducial.truth.shape, (10, 5))
        self.assertTrue(np.isfinite(result.evaluation.log_prob).all())
        self.assertEqual(result.fit.split.test_cube_ids.tolist(), results['void'].fit.split.test_cube_ids.tolist())
        directory = self.root / 'out'
        result.save_diagnostics(directory)
        for name in ('summary.csv', 'calibration.csv', 'fiducial_summary.csv', 'fisher.csv', 'metadata.json',
                     'scores.json', 'diagnostics.npz', 'model.pt'):
            self.assertTrue((directory / name).is_file(), name)
        model = TrainedPosterior.load(directory / 'model.pt')
        x = result.fit.split.x_test[0]
        a = model.sample(x, 32, seed=3)
        b = result.fit.sample(x, 32, seed=3)
        torch.testing.assert_close(a, b)
        import matplotlib
        matplotlib.use('Agg')
        from src import SBIPlotter
        plotter = SBIPlotter()
        for fig in (*plotter.diagnostics(result), plotter.calibration(result.evaluation), plotter.sbc(result.evaluation),
                    plotter.tarp(result.evaluation), plotter.posterior(result, n_samples=500),
                    plotter.fiducial_posterior(result, result.fiducial_x[0], n_samples=500),
                    plotter.fiducial_scatter(result), plotter.compare_errors(results),
                    plotter.compare_metrics(pipeline.compare(results)), plotter.correlation(result.fisher),
                    plotter.fisher_ellipses({k: r.fisher for k, r in results.items()})):
            self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()
