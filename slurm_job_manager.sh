#!/bin/bash
#
# SLURM Throttling Submission Script for SNARE.py
#
# --- USER CONFIGURATION ---
MAX_RUNS=20            # <<< 1. Total number of simulation runs you want to perform (X)
MAX_CONCURRENT=464     # <<< 2. Your maximum allowed concurrent running/pending jobs
SLEEP_INTERVAL=60      # <<< 3. Time (in seconds) to wait between job limit checks
USERNAME="cfonsecahenrique" # <<< 4. Your username for job counting

# --- SLURM JOB TEMPLATE SETUP ---
# Create a temporary file for the actual SLURM job
JOB_SCRIPT_TEMPLATE="snares_job_template.sh"

# Note: Adjust the resources below to match the typical needs of SNARE.py
cat > ${JOB_SCRIPT_TEMPLATE} << EOF
#!/bin/bash
#SBATCH --job-name=SNARE_run
#SBATCH --output=logs/snares_%A_%a.out
#SBATCH --error=logs/snares_%A_%a.err
#SBATCH --time=01:00:00        # Max time for each run (e.g., 1 hour)
#SBATCH --ntasks=1             # Number of tasks
#SBATCH --cpus-per-task=1      # Number of CPU cores per task
#SBATCH --mem=4G               # Memory per task
#SBATCH --partition=LONG       # <--- NEW: Specifies the LONG queue/partition

# Load necessary modules (e.g., Python environment)
# module load python/3.10

echo "Starting SNARE run: \$1"
# Execute the Python program with the specified input file
python SNARE.py \$1
echo "Finished SNARE run: \$1"
EOF
# --- END SLURM JOB TEMPLATE SETUP ---


# --- FUNCTION TO REPORT LOAD ---
# This function is called when the submission script needs to wait (throttle).
report_cluster_load () {
    echo "--- 📊 Cluster Load Impact Report ---"

    # Get the count of your currently RUNNING (R) or PENDING (PD) jobs
    CURRENT_JOBS=$(squeue -h -u ${USERNAME} -t R,PD -o %i | wc -l)

    # Get total number of allocated/available cores on the cluster (nodes in R/A state)
    TOTAL_AVAIL_CORES=$(sinfo -h -o "%c" -t idle,alloc | awk '{s+=$1} END {print s}')
    # Get your currently requested cores (R state only, as PENDING jobs don't consume cores yet)
    YOUR_ALLOC_CORES=$(squeue -u ${USERNAME} -t R -o "%C" -h | awk '{s+=$1} END {print s}')

    echo "Your currently RUNNING/PENDING jobs: ${CURRENT_JOBS} / ${MAX_CONCURRENT} max."
    echo "Your currently requested CPU cores (Running jobs): **${YOUR_ALLOC_CORES} cores**"
    echo "Total CPU cores allocated/available on the cluster: **${TOTAL_AVAIL_CORES} cores**"

    if [[ ${TOTAL_AVAIL_CORES} -gt 0 ]]; then
        # Use 'bc' for floating point division
        LOAD_PERCENT=$(echo "scale=2; (${YOUR_ALLOC_CORES} / ${TOTAL_AVAIL_CORES}) * 100" | bc)
        echo "Your current load impact (by core count): **${LOAD_PERCENT}%** of total cluster capacity."
    else
        echo "Cannot calculate load percentage: Total available cores is zero."
    fi
    echo "-----------------------------------"
}
# --- END FUNCTION ---


# --- MAIN SUBMISSION LOGIC ---

# 1. Start Timing
START_TIME=$(date +%s)
echo "🚀 Starting submission process at $(date)..."
echo "Target Runs: ${MAX_RUNS}. Concurrent Limit: ${MAX_CONCURRENT}."

# Ensure a logs directory exists
mkdir -p logs

for run_number in $(seq 1 ${MAX_RUNS}); do

    # 2. Concurrency Check (The Throttling Mechanism)
    echo "--- Run ${run_number}/${MAX_RUNS}: Checking concurrency..."

    # Get the count of your currently RUNNING (R) or PENDING (PD) jobs
    CURRENT_JOBS=$(squeue -h -u ${USERNAME} -t R,PD -o %i | wc -l)

    while [[ ${CURRENT_JOBS} -ge ${MAX_CONCURRENT} ]]; do
        echo "🚨 LIMIT REACHED: ${CURRENT_JOBS} jobs running/pending (${MAX_CONCURRENT} max)."
        echo "   Waiting for ${SLEEP_INTERVAL} seconds for a slot to free up..."

        # Call the function to report the load while waiting
        report_cluster_load

        sleep ${SLEEP_INTERVAL}

        # Re-check the job count
        CURRENT_JOBS=$(squeue -h -u ${USERNAME} -t R,PD -o %i | wc -l)
    done

    # 3. Submit the Job
    echo "✅ Submitting job ${run_number}. Current jobs: ${CURRENT_JOBS}."

    # We use the template file and pass the required parameter (inputs/INPUT.yaml)
    sbatch ${JOB_SCRIPT_TEMPLATE} inputs/norm_sweep_allg.yaml

done

echo "🎉 All ${MAX_RUNS} jobs have been submitted or are pending."

# 4. Final Time Calculation
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo "⏱️ Total submission time elapsed: $(($TOTAL_TIME / 60)) minutes and $(($TOTAL_TIME % 60)) seconds."

# Clean up the temporary job script template
rm -f ${JOB_SCRIPT_TEMPLATE}