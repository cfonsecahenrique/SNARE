#!/bin/bash

# --- USER CONFIGURATION ---
USERNAME="cfonsecahenrique"
PROJECT_HOME="/home/${USERNAME}/SNARE" # <-- SET YOUR PROJECT'S HOME PATH
VENV_NAME="snare"
INPUT_FILE="inputs/test.yaml"
OUTPUT_DIR="results" # Directory to store final output in $PROJECT_HOME
LOG_FILE="snares_local_$(date +%Y%m%d_%H%M%S).log"

# Use /tmp and include PID for a unique, disposable work area
LOCAL_WORK_DIR="/tmp/run_$$_${USERNAME}"
mkdir -p "$LOCAL_WORK_DIR" || { echo "Error: Failed to create local directory."; exit 1; }

# Record start time and initial log information
echo "--- SNARE Local Run Script Started at $(date) ---" | tee -a "$LOG_FILE"
echo "Local Working Directory: $LOCAL_WORK_DIR" | tee -a "$LOG_FILE"
echo "Project Home: $PROJECT_HOME" | tee -a "$LOG_FILE"

# --- 1. STAGE DATA (Copy In) ---
echo "Staging files from $PROJECT_HOME to $LOCAL_WORK_DIR" | tee -a "$LOG_FILE"

# Copy all code and inputs from $PROJECT_HOME recursively
cp -r "${PROJECT_HOME}"/*.py "$LOCAL_WORK_DIR/" || { echo "Error: Failed to copy Python files."; exit 1; }
cp -r "${PROJECT_HOME}/inputs" "$LOCAL_WORK_DIR/" || { echo "Error: Failed to copy inputs directory."; exit 1; }

# Change directory to the local workspace
cd "$LOCAL_WORK_DIR" || { echo "Error: Failed to change directory to local work space."; exit 1; }

# --- 2. SETUP ENVIRONMENT ---

# Initialize micromamba (using eval is required for shell hooks)
echo "Initializing micromamba..." | tee -a "$LOG_FILE"
eval "$(micromamba shell hook --shell bash)" || { echo "Error: Micromamba initialization failed."; exit 1; }

# Activate the Virtual Environment
echo "Activating micromamba environment: $VENV_NAME" | tee -a "$LOG_FILE"
micromamba activate "$VENV_NAME" || { echo "Error: Failed to activate environment $VENV_NAME."; exit 1; }

# --- 3. EXECUTE PROGRAM (Backgrounded) ---

echo "Starting SNARE.py in the background..." | tee -a "$LOG_FILE"

# The nohup command should direct its output to a specific file, NOT the console.
# We redirect stdout (1) and stderr (2) to a specific run log file within the local directory.
nohup python3 SNARE.py "$INPUT_FILE" > run_output.log 2>&1 &

# Capture the PID of the backgrounded job
RUN_PID=$!
echo "SNARE.py is running with PID: $RUN_PID" | tee -a "$LOG_FILE"

# 4. MONITOR AND WAIT (Optional, but recommended)
echo "Waiting for background process (PID $RUN_PID) to finish..." | tee -a "$LOG_FILE"
# Wait for the specific background job to complete
wait $RUN_PID

# Check the exit status of the python job
JOB_STATUS=$?

echo "SNARE.py finished with exit status: $JOB_STATUS" | tee -a "$LOG_FILE"

# --- 5. STAGE OUT DATA (Copy Back) ---

# Deactivate venv BEFORE copying back, as it's not strictly necessary afterward.
echo "Deactivating environment." | tee -a "$LOG_FILE"
micromamba deactivate

# Create the final destination directory if it doesn't exist
mkdir -p "${PROJECT_HOME}/${OUTPUT_DIR}"

if [ "$JOB_STATUS" -eq 0 ]; then
    echo "Copying results back to NFS..." | tee -a "$LOG_FILE"
    # Copy relevant results and the run log back to permanent storage
    cp run_output.log "${PROJECT_HOME}/${OUTPUT_DIR}/run_output_${RUN_PID}.log"
    # Assuming the SNARE.py produces a file called final_data.json
    # cp -r *.csv "${PROJECT_HOME}/${OUTPUT_DIR}/"
else
    echo "Job failed (Status $JOB_STATUS). Copying logs for debugging." | tee -a "$LOG_FILE"
    cp run_output.log "${PROJECT_HOME}/${OUTPUT_DIR}/failed_log_${RUN_PID}.log"
fi

# --- 6. CLEAN UP LOCAL TEMP DIRECTORY ---

echo "Cleaning up local directory: $LOCAL_WORK_DIR" | tee -a "$LOG_FILE"
rm -rf "$LOCAL_WORK_DIR"

echo "--- SNARE Local Run Script Finished ---" | tee -a "$LOG_FILE"
