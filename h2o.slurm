#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name=h2o_flight_prediction
#SBATCH --time=0-08:00:00         # format: D-HH:MM:SS (8 hours for H2O AutoML)
#SBATCH --partition=CPUQ          # CPU partition (H2O works well on CPU)
#SBATCH --mem=96G                 # Asking for 96GB RAM (H2O is memory intensive)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32        # Request 32 CPU cores for H2O cluster
#SBATCH --output=out_h2o.txt      # Specifying 'stdout'
#SBATCH --error=out_h2o.err       # Specifying 'stderr'
#SBATCH --mail-user=afrasum@stud.ntnu.no
#SBATCH --mail-type=ALL

echo "Starting H2O.ai flight prediction job"

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Load Python and create completely isolated virtual environment
module load Python/3.11.3-GCCcore-12.3.0
python -m venv --clear h2o_venv
source h2o_venv/bin/activate

# Ensure no system packages interfere
export PYTHONNOUSERSITE=1
export PIP_NO_DEPS=false
export PIP_IGNORE_INSTALLED=true

# Upgrade pip
pip install --upgrade pip

# Install core scientific computing libraries first
pip install --no-cache-dir numpy pandas scikit-learn

# Install H2O.ai with Java dependencies
pip install --no-cache-dir h2o

# Install additional ML libraries that H2O might use
pip install --no-cache-dir matplotlib seaborn

# Set environment variables for H2O cluster optimization
export H2O_CLUSTER_NAME="avinor_h2o_cluster"
export H2O_LOG_LEVEL="INFO"

# Set multiprocessing environment for optimal performance
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

# Java optimization for H2O
export _JAVA_OPTIONS="-Xms8g -Xmx80g -XX:+UseG1GC -XX:+UseStringDeduplication"

# H2O specific environment variables
export H2O_CLUSTER_NAME="avinor_hpc_cluster"
export H2O_CLOUD_NAME="avinor_cloud"

echo "Environment setup complete:"
echo "  SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "  SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE}"
echo "  Java Options: ${_JAVA_OPTIONS}"

# Run the H2O script
echo "Starting H2O AutoML training with ${SLURM_CPUS_PER_TASK:-32} CPUs..."
cd prediction
python h2o_pipeline.py

echo "Installed packages:"
pip freeze

echo "H2O.ai job completed"