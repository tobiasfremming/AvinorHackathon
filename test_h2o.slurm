#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name=test_h2o_setup
#SBATCH --time=0-00:30:00         # format: D-HH:MM:SS (30 minutes for testing)
#SBATCH --partition=CPUQ          # CPU partition
#SBATCH --mem=16G                 # Asking for 16GB RAM for testing
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8         # Request 8 CPU cores for testing
#SBATCH --output=out_h2o_test.txt # Specifying 'stdout'
#SBATCH --error=out_h2o_test.err  # Specifying 'stderr'
#SBATCH --mail-user=afrasum@stud.ntnu.no
#SBATCH --mail-type=ALL

echo "Starting H2O.ai setup test job"

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Load Python and create test virtual environment
module load Python/3.11.3-GCCcore-12.3.0
python -m venv --clear h2o_test_venv
source h2o_test_venv/bin/activate

# Ensure no system packages interfere
export PYTHONNOUSERSITE=1
export PIP_NO_DEPS=false
export PIP_IGNORE_INSTALLED=true

# Upgrade pip
pip install --upgrade pip

# Install H2O and dependencies
pip install --no-cache-dir numpy pandas scikit-learn h2o

# Set environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Java optimization for H2O (conservative for testing)
export _JAVA_OPTIONS="-Xms2g -Xmx12g -XX:+UseG1GC"

echo "Environment setup complete for testing"

# Run the H2O test script
echo "Running H2O setup verification..."
cd prediction
python test_h2o_setup.py

echo "H2O setup test completed"