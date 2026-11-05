import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src import (MomentEvaluator, MomentTrainer, MomentTrainingConfig,
                 SpectrumConfig, SpectrumDataset)
from src.pipeline import ExperimentResult


class MomentNetworkTest(unittest.TestCase):
    def test_two_stage_moments_on_informative_synthetic_data(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(7)
        theta = rng.uniform(-1, 1, (180, 1)).astype('float32')
        x = (theta + rng.normal(0, 0.08, theta.shape)).astype('float32')
        data = SpectrumDataset(x=torch.from_numpy(x), theta=torch.from_numpy(theta),
                               cube_ids=np.arange(len(x)), envs=('synthetic',),
                               k={'synthetic': np.array([0.1])},
                               feature_slices={'synthetic': slice(0, 1)},
                               config=SpectrumConfig(), parameter_names=('theta',))
        config = MomentTrainingConfig(n_train=150, hidden_features=16, hidden_layers=2,
                                      batch_size=32, learning_rate=0.002, max_epochs=120,
                                      patience=15, device='cpu')
        fit = MomentTrainer(config).fit(data)
        means, stds = fit.predict_moments(fit.split.x_test)
        truth = fit.split.theta_test
        self.assertLess(torch.sqrt(torch.mean((means - truth) ** 2)).item(), 0.2)
        self.assertTrue(torch.isfinite(stds).all().item())
        self.assertTrue((stds > 0).all().item())
        self.assertTrue(set(fit.split.train_cube_ids).isdisjoint(set(fit.split.test_cube_ids)))
        self.assertEqual(fit.split.test_cube_ids.tolist(), data.split(150, 42).test_cube_ids.tolist())
        result = ExperimentResult(fit, MomentEvaluator().evaluate(fit))
        self.assertEqual(result.evaluation.interval_method, 'gaussian_from_moments')
        with tempfile.TemporaryDirectory() as directory:
            result.save_diagnostics(directory)
            metadata = json.loads((Path(directory) / 'metadata.json').read_text())
            self.assertEqual(metadata['method'], 'mn')
            self.assertEqual(metadata['interval_method'], 'gaussian_from_moments')
            self.assertEqual(metadata['variance_targets'], 'five_fold_out_of_fold_squared_residuals')
            with np.load(Path(directory) / 'diagnostics.npz') as diagnostics:
                self.assertEqual(set(diagnostics['oof_fold']), set(range(5)))
                self.assertEqual(diagnostics['oof_means'].shape, (150, 1))
                np.testing.assert_allclose(diagnostics['oof_squared_residuals'],
                                           (fit.split.theta_train.numpy() - diagnostics['oof_means']) ** 2,
                                           rtol=1e-5, atol=1e-6)
            self.assertEqual(len(metadata['oof_best_epochs']), 5)
            self.assertEqual(len(result.summary), 1)


    def test_held_out_target_cannot_change_its_oof_prediction(self):
        rng = np.random.default_rng(11)
        theta = rng.uniform(-1, 1, (30, 1)).astype('float32')
        x = (theta + rng.normal(0, 0.1, theta.shape)).astype('float32')

        def dataset(values):
            return SpectrumDataset(x=torch.from_numpy(x.copy()), theta=torch.from_numpy(values.copy()),
                                   cube_ids=np.arange(len(x)), envs=('synthetic',),
                                   k={'synthetic': np.array([0.1])},
                                   feature_slices={'synthetic': slice(0, 1)},
                                   config=SpectrumConfig(), parameter_names=('theta',))

        config = MomentTrainingConfig(n_train=25, hidden_features=8, batch_size=8,
                                      max_epochs=3, patience=2, device='cpu')
        original = dataset(theta)
        first_fit = MomentTrainer(config).fit(original)
        changed = theta.copy()
        changed[first_fit.split.train_rows[0], 0] += 10
        second_fit = MomentTrainer(config).fit(dataset(changed))
        self.assertEqual(first_fit.oof_fold.tolist(), second_fit.oof_fold.tolist())
        self.assertTrue(torch.equal(first_fit.oof_means[0], second_fit.oof_means[0]))


if __name__ == '__main__':
    unittest.main()
