#! /bin/bash

#SBATCH --job-name=1024PostImpressionistImageGenTraining
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu[002]

# source /home/s25vargason1/miniconda3/bin/activate
source /home/s25vargason1/miniconda3/etc/profile.d/conda.sh
conda activate venv

# srun -- unbuffered python 'Post-Impressionism StyleGAN2-ADA Training.py'

srun python slurm_1024_post_impressionist.py


# name that will show up in the queue(two dashes)
# the partitions to run in GPU(two dashes)
# node in the GPU partition(two dashes)

# directory of /miniconda3/bin/activate
# activate your virtual environment

# run your python code file(two dashes)