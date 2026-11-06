# quijotes-sbi

Simulation-based inference of `Omega_m, Omega_b, h, n_s, sigma8` from cosmic-web
environment statistics of Quijote FoF halos (z = 0.5, 1 (Gpc/h)³) classified with
ASTRA. It follows GIGANTES ([Kreisch et al. 2022](https://iopscience.iop.org/article/10.3847/1538-4357/ac7d4b)):
a **Fisher forecast** from the fiducial and derivative sets, and **neural
inference** trained on the latin hypercube (NPE ensembles and Moment Networks).

## Data

| Product | LH | Fiducial | Derivatives (14 sets) |
| --- | --- | --- | --- |
| `pk`: real-space P0(k) for `all`, `void`, `sheet`, `filament`, `knot`, `random_void` | 2000 | 2000 | 500 each |
| `pk_rsd`: redshift-space P0, P2, P4 | **54–84** | 2000 | 500 each |
| `astra`: per-halo environment probabilities | 2000 | 2000 | 500 each |
| `hmf`: FoF halo masses + ASTRA labels | 2000 | 2000 | 500 each |

Derivative sets: `Om`, `Ob2`, `h`, `ns`, `s8` (±), plus `fiducial_ZA` and
`Mnu_p/pp/ppp` for Mν (Fisher only; the LH has massless neutrinos). Run `python -m src inventory` to check completeness.

**Validation.** A file is rejected, with its path in the error, if any of these holds:
- its metadata (set, realization, snapshot, box, sample, space, z) does not match its location;
- its `omega_m` differs from the parameter file;
- the shot noise is inconsistent;
- the ASTRA probabilities are malformed;
- the FoF particle mass does not match Ω_m.

Across products, the counts must agree in every realization:
- environment counts sum to the full catalog;
- ASTRA argmax counts equal the P(k) `n_objects`;
- the FoF halo count equals the number of ASTRA rows.

All 128 tables currently pass.


## Experiments

An experiment concatenates blocks `statistic:sample`. `envs` expands to the four
environments, and a bare sample name (`void`) means `pk:void`.

| Block | Features |
| --- | --- |
| `pk:<sample>` | log10 P0(k), `--kmin` ≤ k ≤ `--kmax` (39 bins by default) |
| `pkrsd:<sample>` | log10 P0 and P_l/P0 (`--rsd-multipoles`) |
| `nbar:<sample>` | log10 number of objects |
| `hmf:<sample>` | log10(1+N) halos per mass bin (`--hmf-edges`, lowest edge above the 20-particle limit) |
| `astra:<env>` | log10(1+N) halos per ASTRA-probability bin (`--astra-bins`) |


## Inference

- **NPE** (`--method npe`): an equal-weight mixture of `--ensemble` (default 5)
  independently trained flows (`--density-estimator nsf|maf|mdn`).
  - `--embedding-dim N` adds a jointly trained MLP embedding.
  - `--compression moped` replaces the features with score-compressed summaries
    built only from the fiducial and derivative sets. These are optimal near the
    fiducial and lossy far from it.
- **Moment Networks** (`--method mn`, or `both`): marginal means and standard
  deviations, using 5-fold out-of-fold residuals (Jeffrey & Wandelt 2020).
  Intervals assume Gaussian marginals.

## Validation metrics

| Metric | Meaning |
| --- | --- |
| `bias`, `RMSE`, `R2` | accuracy of the posterior mean |
| `coverage_68/95`, `calibration_error` | central-interval coverage (19 levels) vs nominal |
| `pit_ks_pvalue`, `chi2` | rank (SBC) uniformity; mean squared z-score (≈ 1) |
| `std_over_prior` | posterior std / prior std (1 = uninformative) |
| `*_interior` | metrics away from the prior edges |
| `nll` | −log q(θ_true \| x): proper score, lower is better |
| `tarp_max_deviation` | joint-posterior calibration (TARP, Lemos et al. 2023) |

`--fiducial N` evaluates the posterior on N fiducial realizations and compares
it with Fisher. This catches systematic biases or processing differences between
sets. Its coverage is frequentist at a single point: for prior-dominated
parameters (h, Ω_b) it is close to 1 by construction. Bayesian calibration is
measured on the test set.

## Fisher

`python -m src fisher` uses exactly the NPE features and reports three
marginalized errors:
- `sigma_fisher`: standard, with the Hartlap correction. Optimistic when the derivatives are noisy.
- `sigma_fisher_compressed`: conservative (split-derivative compression).
- `sigma_fisher_combined`: the geometric mean of the two ([Coulton & Wandelt 2023](https://arxiv.org/abs/2305.08994)).

`--convergence` varies the number of covariance and derivative realizations, and
`--fisher-params ... Mnu` adds neutrino mass. With the current data the
covariance is converged, but the derivatives are not for Ω_b, h and n_s. Treat
Fisher as a reference for those parameters; Ω_m and σ8 are robust.

## Outputs

Each experiment directory contains:
- metrics: `summary.csv`, `calibration.csv`, `scores.json`, `fiducial_summary.csv`, `fisher.csv`;
- `diagnostics.npz`: per-cube arrays and train/test IDs;
- `metadata.json`: full configuration, ensemble training history, git commit and versions;
- `model.pt`: the trained model;
- with `--plots`: the figures.

Runs with several experiments add `comparison.csv`, `fiducial_comparison.csv`
and comparison figures. Results are saved as each experiment finishes.

## Python API

```python
from src import QuijoteSummaryLoader, TrainingConfig, SBITrainer, FisherForecaster, SBIPipeline, TrainedPosterior

loader = QuijoteSummaryLoader()
pipeline = SBIPipeline(loader, SBITrainer(TrainingConfig(n_train=1800, ensemble_size=5, device='cuda')),
                       fisher=FisherForecaster(loader), n_fiducial=500)
results = pipeline.run_many({'void': 'pk:void', 'envs': 'pk:envs'})
print(SBIPipeline.compare(results))
results['envs'].save_diagnostics('results/envs')

model = TrainedPosterior.load('results/envs/model.pt')
fid = loader.load_set('fiducial', 'pk:envs')
samples = model.sample(fid.x_tensor[0], n_samples=10_000, seed=1)
```

Modules:
- `config`: constants and configuration;
- `fits`, `products`, `store`: validated readers and the cache;
- `features`: blocks and datasets;
- `fisher`: Fisher forecasts;
- `inference`, `moments`: NPE and Moment Networks;
- `evaluation`: metrics;
- `pipeline`, `plotting`, `__main__`: orchestration, figures and the command line.

The original `QuijoteDataLoader` is kept for the notebooks.

## Tests

```bash
python -m unittest discover -s tests -v
QUIJOTE_RUN_SBI_SMOKE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v   # + end-to-end NPE
```

The tests build a synthetic Quijote tree with the real file formats.