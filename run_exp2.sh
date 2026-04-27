#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --account=icompagn
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

set -euo pipefail

cd "$(dirname "$0")"
module load python/3.11.7
source .venv/bin/activate

mkdir -p run_logs
export AP4FED_QUIET=1
ROUNDS="${AP4FED_ROUNDS:-2}"

python build_paper_experiments_500clients_k5.py --quiet --rounds "$ROUNDS" \
  > /dev/null \
  2> "run_logs/exp2_${SLURM_JOB_ID:-manual}.err"
