#!/bin/bash
#SBATCH --job-name=v1_111664
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/$USER/modelv1/output/%x_%j.out
#SBATCH --error=/scratch/$USER/modelv1/output/%x_%j.err

set -x	#Enable shell debug mode

echo "Accesing virtual python environment..."
source $HOME/venvs/BaIP2/bin/activate

echo "Making output directory..."
mkdir -p /scratch/$USER/modelv1/output/$SLURM_JOB_ID

echo "Moving and Extracting data..."
cp /scratch/$USER/kittiYOLOData.tar.gz $TMPDIR
cd $TMPDIR
echo "Current working directory is..."
pwd
ls -l
tar -xzf kittiYOLOData.tar.gz
cd kittiYOLOData
pwd
ls -l

echo "Starting training..."
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=10 imgsz=640 batch=64 workers=16 device=0

echo "Saving Results..."
cp -r runs /scratch/$USER/modelv1/output/$SLURM_JOB_ID

