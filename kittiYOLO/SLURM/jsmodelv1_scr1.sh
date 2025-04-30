#!/bin/bash
#SBATCH --job-name=v1_1_16
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4000
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/BaIP2/bin/activate

echo "Making output directory..."
mkdir -p /scratch/$USER/modelv1/output

echo "Moving and Extracting data..."
cp /scratch/$USER/kittiYOLOdata.tar.gz $TMPDIR
cd $TMPDIR
tar -xzf kittiYOLOdata.tar.gz
cd kittiYOLOData

echo "Starting training..."
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=10 imgsz=640 batch=64 workers=16 device=0

echo "Saving Results..."
mkdir -p /projects/$USER/modelv1/output
cp -r runs /projects/$USER/modelv1/output/

echo "Job finished successfully."
