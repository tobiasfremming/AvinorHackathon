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

# Load Python and create virtual environment
module load Python/3.11.3-GCCcore-12.3.0
python -m venv autogluon_venv
source autogluon_venv/bin/activate
pip install --upgrade pip

# Install AutoGluon first (picky about dependency versions)
pip install autogluon.tabular

# Install any additional requirements if they exist
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install any remaining packages that might be needed
pip install pyproject-toml

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