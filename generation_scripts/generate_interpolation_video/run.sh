#! /bin/bash

#SBATCH --job-name=generate_interpolation-video
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu[001]

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

source ~/miniconda3/bin/activate
conda activate venv

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/s25vargason1/generation_scripts/generate_interpolation_video

srun -u python generate_interpolation_video.py

# name that will show up in the queue(two dashes)
# the partitions to run in GPU(two dashes)
# node in the GPU partition(two dashes)

# directory of /miniconda3/bin/activate
# activate your virtual environment

# run your python code file(two dashes)