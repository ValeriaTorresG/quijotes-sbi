# quijotes-sbi

Clases en `src/` para reutilizar el flujo de `sbi_env.ipynb`, `sbi_all.ipynb` y
`sbi_init.ipynb`: lectura de espectros de potencia de los cubos Latin hypercube,
SBI por ambiente y SBI con espectros concatenados.

Usa Python >= 3.10 con las dependencias de `requirements.txt`. El código se ha
verificado con el entorno del kernel **Python (sbi)** de los notebooks y sbi 0.27.
Ejecuta los ejemplos desde la raíz del repositorio para poder importar `src`.

```bash
python -m pip install -r requirements.txt
```

## Ejecutar desde la terminal

Desde la raíz del repositorio y con el entorno `sbi` activo:

```bash
source /pscratch/sd/v/vtorresg/venvs/sbi/bin/activate

# Un ambiente: 800 cubos, 720 para train y 80 para test.
python -m src --envs void --plots

# Un único modelo con los cuatro ambientes concatenados.
python -m src --envs void sheet filament knot --plots

# Cada ambiente por separado y además la combinación, con los mismos IDs train/test.
python -m src --envs void sheet filament knot --compare --plots

# Comprobación breve del flujo (no es un entrenamiento para resultados científicos).
python -m src --envs void --n-cubes 24 --n-train 20 \
    --max-epochs 2 --hidden-features 16 --num-transforms 2 \
    --batch-size 8 --n-samples 16 --output results/smoke

python -m src --help
```

Los comandos usan las rutas de datos de los notebooks; se pueden cambiar con
`--base` y `--params-file`. Por defecto, guardan `summary.csv`, `diagnostics.npz`
y `metadata.json` en `results/<ambientes>`; `--plots` añade tres figuras PNG.
`--compare` guarda una carpeta por modelo y `comparison.csv` bajo
`results/comparison`. Puedes elegir otro destino con `--output`.
Estos archivos contienen diagnósticos y configuración; el posterior entrenado
se mantiene en memoria durante la ejecución. Para usarlo después del ajuste
en un notebook, utiliza `pipeline.run(...)` y `result.posterior`.
Ejecuta los entrenamientos completos en un nodo de cómputo.

## Un ambiente o una combinación

```python
from pathlib import Path
from src import (
    QuijoteDataLoader, SpectrumConfig, TrainingConfig, EvaluationConfig,
    SBITrainer, PosteriorEvaluator, SBIPipeline, SBIPlotter,
)

loader = QuijoteDataLoader(
    base=Path('/pscratch/sd/v/vtorresg/quijotes/PowerSpectrum/FoF/latin_hypercube'),
    params_file=Path('latin_hypercube_params.txt'),
    config=SpectrumConfig(kmin=0.01, kmax=0.5, transform='log10'),
)
trainer = SBITrainer(TrainingConfig(n_train=720, split_seed=42, train_seed=42))
evaluator = PosteriorEvaluator(EvaluationConfig(n_samples=5000, sample_seed=123))
pipeline = SBIPipeline(loader, trainer, evaluator)

# Un modelo condicionado en Pk de void.
result = pipeline.run(envs='void', cube_ids=range(800))
print(result.summary.to_string(index=False))

# Otro modelo condicionado en la concatenación de cuatro espectros.
combined = pipeline.run(
    envs=['void', 'sheet', 'filament', 'knot'], cube_ids=range(800),
)
print(combined.summary)

# Figuras reutilizables, sin imponer LaTeX ni modificar rcParams.
plotter = SBIPlotter()
fig_prediction, fig_coverage = plotter.diagnostics(combined)
fig_posterior = plotter.posterior(combined, idx=0, n_samples=50_000)
fig_prediction.savefig('prediction.png', dpi=200, bbox_inches='tight')
# En un notebook, puedes terminar la celda con fig_prediction o usar plt.show().

# Exporta métricas, intervalos, IDs train/test y configuración; no guarda el modelo.
combined.save_diagnostics('results/combined')
```

`all` es el espectro del catálogo completo de halos. Para combinar ambientes se
pasa una lista: se concatenan las características, en ese orden, y se entrena un
único modelo NPE. Los ambientes disponibles son `all`, `void`, `sheet`,
`filament`, `knot` y `random_void`.

## Comparar ambientes con los mismos cubos

`run_many` lee cada ambiente una sola vez y reutiliza sus características. Entrena
un modelo independiente por entrada del diccionario. La selección train/test se
hace por ID de cubo, usando `n_train` y `split_seed`, y coincide entre todos los
experimentos para los mismos IDs, incluso si se pasan en otro orden.

```python
results = pipeline.run_many(
    experiments={
        'all': 'all',
        'void': 'void',
        'sheet': 'sheet',
        'filament': 'filament',
        'knot': 'knot',
        'random_void': 'random_void',
        'environments': ['void', 'sheet', 'filament', 'knot'],
        'all_and_environments': ['all', 'void', 'sheet', 'filament', 'knot'],
    },
    cube_ids=range(800),
)
comparison = pipeline.compare(results)
print(comparison)
comparison.to_csv('comparison.csv', index=False)
for name, experiment in results.items():
    experiment.save_diagnostics(Path('results') / name)
```

La evaluación devuelve bias, RMSE, desviación posterior media, ancho medio del
intervalo de 68 % y cobertura de los intervalos centrales de 68 % y 95 %, igual
que los notebooks. `n_train=720` deja 80 cubos de prueba de un total de 800;
`validation_fraction=0.1` reserva además parte de esos 720 para validación interna
de SBI. Para reproducir la partición de `sbi_all.ipynb`, usa `n_train=450`.

## Usar las clases por separado

```python
# Leer una sola vez y elegir subconjuntos de ambientes en memoria.
data = loader.load(['all', 'void', 'sheet', 'filament', 'knot'], range(800))
void_data = data.select('void')
env_data = data.select(['void', 'sheet', 'filament', 'knot'])

# x: (n_cubos, n_características); theta: (n_cubos, 5).
print(env_data.x.shape, env_data.theta.shape)
print(env_data.k['void'], env_data.feature_slices['void'])

fit = trainer.fit(env_data)
evaluation = evaluator.evaluate(fit)
print(evaluation.summary)
print(fit.split.train_cube_ids, fit.split.test_cube_ids)

# x debe tener el mismo orden de ambientes, bins y transformación del entrenamiento.
samples = fit.sample(fit.split.x_test[0], n_samples=5000, seed=123)
posterior = fit.posterior  # Objeto SBI para otros análisis.
```

| Clase | Responsabilidad |
| --- | --- |
| `SpectrumConfig`, `TrainingConfig`, `PriorConfig`, `EvaluationConfig` | Configuración explícita y validada |
| `NPZSpectrumReader`, `TextSpectrumReader` | Lectura de archivos sin entrenamiento |
| `QuijoteDataLoader` | Selección de bins, transformación y alineación con los parámetros |
| `SpectrumDataset`, `DatasetSplit` | Datos, bloques por ambiente y partición por cubo |
| `SBITrainer`, `FitResult` | NPE con NSF, prior y muestreo posterior |
| `PosteriorEvaluator`, `EvaluationResult` | Evaluación de los cubos de prueba |
| `SBIPipeline`, `ExperimentResult` | Ejecución, comparación y exportación de diagnósticos |
| `SBIPlotter` | Figuras de predicción, cobertura y posteriores marginales |

El orden de parámetros es `Omega_m`, `Omega_b`, `h`, `n_s`, `sigma8`. Los límites
del prior conservan los valores de tus notebooks y se pueden sustituir mediante
`SBITrainer(config=..., prior=PriorConfig(low=..., high=...))`.

## Otros archivos y espectros de materia

Para otros snapshots, nombres de archivo o estimadores, pasa `env_files` y un
lector. `NPZSpectrumReader(k_key='k_center', pk_key='Pk0')` permite cambiar las
claves del NPZ. Un lector propio solo necesita implementar `read(path)` y devolver
`PowerSpectrum(k, pk, bin_low=None, bin_high=None)`.

```python
from src import TextSpectrumReader

matter_loader = QuijoteDataLoader(
    base='/pscratch/sd/v/vtorresg/quijotes/Pk/matter/latin_hypercube',
    params_file='latin_hypercube_params.txt',
    env_files={'matter': 'Pk_m_z=0.5.txt'},
    reader=TextSpectrumReader(),
)
matter_pipeline = SBIPipeline(
    matter_loader, SBITrainer(TrainingConfig(n_train=1800)), evaluator,
)
# Equivalente al flujo de sbi_init.ipynb, con 2000 cubos explícitos.
# matter_result = matter_pipeline.run('matter', range(2000))
```

El loader alinea `theta` con `params[cube_id]`, verifica las mallas de `k` y bordes
entre cubos de cada ambiente y rechaza IDs repetidos o fuera del archivo de
parámetros. Las mallas pueden diferir entre ambientes. Rechaza valores no finitos
y `Pk <= 0` en los bins seleccionados cuando se usa `log10`; usa
`SpectrumConfig(transform='none')` para conservar `Pk` sin logaritmo.

## Verificación

```bash
python -m unittest discover -s tests -v
# Prueba breve adicional con NPE real, sin entrenar los modelos de 800 cubos:
QUIJOTE_RUN_SBI_SMOKE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Las semillas separan la partición, el entrenamiento y el muestreo. La
reproducibilidad numérica también depende de las versiones y del dispositivo.
Los valores por defecto conservan CPU y los hiperparámetros de los notebooks;
puedes configurar `device='cuda'` en un nodo que disponga de GPU. Ejecuta los
entrenamientos completos en un nodo de cómputo.

## Segunda versión: Moment Networks

La opción `--method mn` entrena dos redes densas consecutivas con el mismo lector,
los mismos espectros transformados y la misma partición por ID de cubo que NPE.
La primera estima la media marginal `F(x)`. Para entrenar la segunda se
dividen **solo los cubos de entrenamiento** en cinco partes: cada red de medias
se ajusta con cuatro partes y predice la quinta. El residuo cuadrático de cada
cubo se calcula con una predicción hecha sin entrenar con ese cubo. La red de
varianza aprende estos residuos y la red de medias final se reentrena con todos
los cubos de entrenamiento. Es una adaptación de la jerarquía de [Jeffrey y
Wandelt (2020), ecuaciones 4 y 5](https://arxiv.org/pdf/2011.05991), usada
como método en la [aplicación de GIGANTES](https://arxiv.org/pdf/2107.02304)
(sección 5).

```bash
# MN con un ambiente, usando los mismos 800 cubos y 720 para ajuste que NPE.
python -m src --method mn --envs void --n-cubes 800 --n-train 720 --plots

# Entrena ambos métodos sobre los mismos datos y escribe comparison.csv.
python -m src --method both --envs void sheet filament knot \
    --n-cubes 800 --n-train 720 --compare --plots
```

`--hidden-features` controla el ancho de ambas redes y `--hidden-layers` su
profundidad (por defecto 128 y 2); `--num-transforms` solo aplica a NPE. Cada
ajuste elige el número de épocas mediante validación interna y luego
reentrena con todos sus cubos disponibles. Cada fold ajusta la normalización
solo con sus cuatro partes; los modelos finales la ajustan con todos los cubos
de entrenamiento, sin usar los cubos de prueba. Los residuos se forman en
unidades físicas y la red de varianza los normaliza por la escala del parámetro
al cuadrado. `diagnostics.npz` guarda `oof_means`, `oof_fold` y
`oof_squared_residuals` en el orden de `train_cube_ids`; `metadata.json`
registra las cinco particiones y las épocas seleccionadas. Los resultados MN se guardan bajo `results/mn_<ambientes>` o,
en una comparación de métodos, bajo `results/method_comparison/mn/`.

```python
from src import (MomentTrainer, MomentTrainingConfig, MomentEvaluator,
                 SBIPipeline)

mn_pipeline = SBIPipeline(loader, MomentTrainer(MomentTrainingConfig(n_train=720)),
                          MomentEvaluator())
mn_result = mn_pipeline.run('void', range(800))
mean, std = mn_result.fit.predict_moments(mn_result.fit.split.x_test[0])
```

MN entrega medias y desviaciones **marginales**, sin muestras ni densidad de la
posterior conjunta. Para mantener las columnas de diagnóstico, los intervalos
de 68 % y 95 % de MN se calculan como `media ± z * std` asumiendo marginales
Gaussianas; `metadata.json` los identifica como `gaussian_from_moments`. Sus
coberturas son una comprobación de esa aproximación y no cuantiles producidos
por la red. Las figuras MN incluyen predicción y cobertura, sin histograma de
posterior. El posterior de NPE sigue evaluándose con muestras. Los residuos
out-of-fold reducen el sesgo optimista por sobreajuste de la red de medias,
pero también incluyen su error de predicción: no garantizan una varianza
posterior exacta. Cada red de fold usa menos ejemplos que la red de medias
final.

La comparación **NPE frente a MN en este repositorio** usa espectros de potencia
`P(k)` de halos. GIGANTES usó la función de tamaños de voids, otro observable,
así que sus errores numéricos no son directamente comparables con los de estos
experimentos. Como en NPE, se guardan diagnósticos y configuración; las redes
entrenadas permanecen en memoria.
