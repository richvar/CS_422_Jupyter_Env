#! /bin/bash

#SBATCH --job-name=512NoFaceYesTransferContinuedLearningPostImpressionistImageGenTraining
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu[002]

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

source ~/miniconda3/bin/activate
conda activate venv

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/s25vargason1/tess/stylegan2-ada-pytorch

srun -u python launch_train_loop.py
