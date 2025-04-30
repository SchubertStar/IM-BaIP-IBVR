#!/bin/bash
#SBATCH --job-name=v1_1_8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4000
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/$USER/modelv1/output
#SBATCH --error=/scratch/$USER/modelv1/output

echo "Loading environment and packages..."
module purge
module load foss/2022a
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/BaIP2/bin/activate

echo "Making directories..."
mkdir -p /scratch/$USER/modelv1/output

echo "Moving and Extracting data..."
cp /scratch/$USER/kittiYOLOdata.tar.gz $TMPDIR
cd $TMPDIR
tar -xzf kittiYOLOdata.tar.gz
cd kittiYOLOData

echo "Starting training..."
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=10 imgsz=640 batch=64 workers=8

echo "Saving Results..."
cp -r runs /projects/$USER/modelv1/output
cp -r /scratch/$USER/modelv1/output /projects/$USER/modelv1/output
