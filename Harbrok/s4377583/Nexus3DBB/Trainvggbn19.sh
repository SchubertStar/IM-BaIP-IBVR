#!/bin/bash
#SBATCH --job-name=VGGbn19
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -x

# Activate Python environment and modules
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load SciPy-bundle/2023.11-gfbf-2023b
module load OpenCV/4.8.1-foss-2023a-CUDA-12.1.1-contrib

source "$HOME/venvs/BaIP3/bin/activate"
export PYTHONPATH=$PWD:$PYTHONPATH

# Prepare output directory
OUTDIR="/scratch/s4377583/Nexus3DBB/3Doutput/${SLURM_JOBID}"
mkdir -p "$OUTDIR"

# Save SLURM job info
cat <<EOF > "$OUTDIR/job_info.txt"
SLURM Job ID:           ${SLURM_JOB_ID}
SLURM Job Name:         ${SLURM_JOB_NAME:-N/A}
Node List:              ${SLURM_NODELIST:-N/A}
Partition:              ${SLURM_JOB_PARTITION:-N/A}
Number of Nodes:        ${SLURM_JOB_NUM_NODES:-N/A}
CPUs per Task:          ${SLURM_CPUS_PER_TASK:-N/A}
Memory per Node:        ${SLURM_MEM_PER_NODE:-N/A}
Time Limit:             ${SLURM_TIMELIMIT:-N/A}
Submit Directory:       ${SLURM_SUBMIT_DIR:-N/A}
EOF

python3 Train.py

# Save results
mv -r /scratch/s4377583/Nexus3DBB/weights "$OUTDIR/weights" 2>/dev/null || echo "Weights not found or failed to copy."



