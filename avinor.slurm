#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name=autogluon_flight_prediction
#SBATCH --time=0-06:00:00         # format: D-HH:MM:SS (6 hours)
#SBATCH --partition=GPUQ          # GPU partition
#SBATCH --mem=64G                 # Asking for 64GB RAM
#SBATCH --nodes=1
#SBATCH --output=outavinor.txt      # Specifying 'stdout'
#SBATCH --error=outavinor.err        # Specifying 'stderr'
#SBATCH --mail-user=afrasum@stud.ntnu.no
#SBATCH --mail-type=ALL

echo "Starting AutoGluon flight prediction job"

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Load Python and create completely isolated virtual environment
module load Python/3.11.3-GCCcore-12.3.0
python -m venv --clear autogluon_venv
source autogluon_venv/bin/activate

# Ensure no system packages interfere
export PYTHONNOUSERSITE=1
export PIP_NO_DEPS=false
export PIP_IGNORE_INSTALLED=true

pip install --upgrade pip

# Install required dependencies first
pip install --no-cache-dir typing_extensions

# Install AutoGluon in isolated environment
pip install --no-cache-dir autogluon.tabular

# Set multiprocessing environment for AutoGluon
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Run the AutoGluon script
echo "Starting AutoGluon training with ${SLURM_CPUS_PER_TASK:-16} CPUs..."
cd prediction
python autogluon.py

pip freeze
echo "AutoGluon job completed"