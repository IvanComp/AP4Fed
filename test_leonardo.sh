#!/bin/bash
#SBATCH --job-name=test_leo
#SBATCH --account=icompagn
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=test_output_%j.txt

set -euo pipefail

cd "$(dirname "$0")"

echo "=== INIZIO TEST SU LEONARDO ==="
echo "Eseguito sul nodo:" $(hostname)
echo "Data e ora:" $(date)

echo ""
echo "--- Test Ambiente Python ---"
module load python
echo "Versione Python di sistema:" $(python --version)

if [ -f ".venv/bin/activate" ]; then
    echo "Attivazione ambiente virtuale..."
    source .venv/bin/activate
    echo "Python in uso:" $(which python)
else
    echo "ATTENZIONE: Nessun ambiente virtuale '.venv' trovato in $(pwd)"
fi

echo ""
echo "=== FINE TEST ==="
