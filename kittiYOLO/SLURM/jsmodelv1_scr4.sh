#!/bin/bash
#SBATCH --job-name=v1_111664
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out      # write .out here
#SBATCH --error=%x_%j.err       # write .err here

set -x  # Enable shell debug mode

echo "Accessing virtual Python environment..."
source "$HOME/venvs/BaIP2/bin/activate"

echo "Creating output directory..."
mkdir -p "/scratch/$USER/modelv1/output/$SLURM_JOB_ID"

echo "Moving and extracting data..."
cp "/scratch/$USER/kittiYOLOData.tar.gz" "$TMPDIR"
cd "$TMPDIR"
tar -xzf kittiYOLOData.tar.gz

# Set working directory to extracted data folder
WORKDIR="$TMPDIR/kittiYOLOData"
cd "$WORKDIR"

echo "Generating dynamic data.yaml with absolute paths..."
cat <<EOF > data.yaml
train: $WORKDIR/images/train
val:   $WORKDIR/images/val

nc: 11
names:
  ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
   'Cyclist', 'Motorcyclist', 'Bus', 'Tram', 'Misc', 'DontCare']
EOF

echo "Starting training..."
yolo task=detect mode=train \
     model=yolov8s.pt \
     data=data.yaml \
     epochs=10 \
     imgsz=640 \
     batch=64 \
     workers=16 \
     device=0

echo "Saving training results..."
cp -r runs "/scratch/$USER/modelv1/output/$SLURM_JOB_ID"

echo "Moving SLURM output files to final directory..."
mv "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "/scratch/$USER/modelv1/output/$SLURM_JOB_ID/"
mv "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "/scratch/$USER/modelv1/output/$SLURM_JOB_ID/"

echo "Job finished successfully."

