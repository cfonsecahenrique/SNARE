#!/bin/bash
#SBATCH --job-name=SNARE
#SBATCH --output=logs/snare_%j.out
#SBATCH --error=logs/snare_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=LONG

YAML="$1"
COMBO="$2"

PROJECT_HOME="/home/cfonsecahenrique/SNARE"
cd "$PROJECT_HOME" || { echo "Cannot cd to $PROJECT_HOME"; exit 1; }

mkdir -p logs outputs

source "$PROJECT_HOME/venv/bin/activate"

echo "[$(date)] YAML=$YAML combo=$COMBO host=$(hostname) cpus=$(nproc)"
python SNARE.py "$YAML" --combo "$COMBO"
STATUS=$?
echo "[$(date)] Done (exit $STATUS)"
exit $STATUS
