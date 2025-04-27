#!/bin/bash
#SBATCH --job-name=1024PostImpressionistImageGenTraining
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu[001]

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Load conda
source ~/miniconda3/bin/activate
conda activate venv

# Fix NumPy compatibility and add library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Go to your project directory
cd /home/s25vargason1/tess/stylegan2-ada-pytorch

# Run training
srun -u python train.py \
  --outdir=training-runs \
  --data=tess/datasets/post-impressionist \
  --gpus=1 \
  --snap=5 \
  --cfg=auto \
  --kimg-per-tick=10 \
  --aug=ada
