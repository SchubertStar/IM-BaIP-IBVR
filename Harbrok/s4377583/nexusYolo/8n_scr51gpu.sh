#!/bin/bash
#SBATCH --job-name=bigv1iyolo8n
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/s4377583/nexusYolo/output/%j/%j.out
#SBATCH --error=/scratch/s4377583/nexusYolo/output/%j/%j.err
set -x

# Activate Python environment
echo "Accessing virtual Python environment..."
source "$HOME/venvs/BaIP2/bin/activate"

# Prepare output directory
OUTDIR="/scratch/$USER/nexusYolo/output/$SLURM_JOB_ID"
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

# Data preparation
echo "Moving and extracting data..."
cp "/scratch/$USER/nexusYolo/finalbig.v1i.yolov8.tar.gz" "$TMPDIR"
cd "$TMPDIR"
tar -xzf finalbig.v1i.yolov8.tar.gz
WORKDIR="$TMPDIR/finalbig.v1i.yolov8"
cd "$WORKDIR"

# Generate data.yaml
echo "Generating dynamic data.yaml..."
cat <<EOF > data.yaml
train: $WORKDIR/images/train
val: $WORKDIR/images/val

nc: 1
names: ['nexusamr']
EOF

# Start training
echo "Starting training..."
yolo task=detect \
     mode=train \
     model=yolov8n.pt \
     data=data.yaml \
     epochs=150 \
     imgsz=320,640 \
     batch=64 \
     workers=16 \
     device=0,1

# Save results
echo "Saving training results..."
cp -r runs/detect/train/weights "$OUTDIR/weights" 2>/dev/null || echo "Weights not found, skipping."
rm -r runs/detect/train/weights
cp -r runs/detect/train "$OUTDIR/train"
