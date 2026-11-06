#!/bin/bash
# Full SBI + Fisher analysis on one Perlmutter GPU node.
#   sbatch -A <account> scripts/run_sbi.slurm            # preset "full"
#   PRESET=environments sbatch -A <account> scripts/run_sbi.slurm
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J quijote-sbi
#SBATCH -o slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source /pscratch/sd/v/vtorresg/venvs/sbi/bin/activate
export OMP_NUM_THREADS=8

PRESET=${PRESET:-full}
OUT=${OUT:-results/${PRESET}_$(date +%Y%m%d_%H%M)}

# 1) Validate every file once (cached; fast on later runs).
python -m src validate --output "$OUT/validation"

# 2) Gaussian reference: Fisher forecast per experiment, incl. convergence tests.
python -m src fisher --preset "$PRESET" --convergence --plots --output "$OUT/fisher"

# 3) NPE ensembles (+ Moment Networks) with test, fiducial and Fisher comparisons.
python -m src train --preset "$PRESET" --method both --ensemble 5 --fisher --fiducial 1000 \
    --plots --output "$OUT/train"
