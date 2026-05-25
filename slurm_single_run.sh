#!/bin/bash
#SBATCH --job-name=SNARE
#SBATCH --output=logs/snare_%j.out
#SBATCH --error=logs/snare_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=LONG

YAML="$1"
COMBO="$2"
RUN="$3"

PROJECT_HOME="/home/cfonsecahenrique/SNARE"
cd "$PROJECT_HOME" || { echo "Cannot cd to $PROJECT_HOME"; exit 1; }

mkdir -p logs outputs

source "$PROJECT_HOME/venv/bin/activate"

echo "[$(date)] YAML=$YAML combo=$COMBO run=$RUN host=$(hostname)"
python SNARE.py "$YAML" --combo "$COMBO" --run "$RUN"
STATUS=$?
echo "[$(date)] Done (exit $STATUS)"
exit $STATUS
