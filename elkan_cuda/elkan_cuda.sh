#!/bin/bash
#SBATCH --job-name=Q3
#SBATCH --output=/scratch/crk239/parallel_proj/elkan_cuda_out.txt
#SBATCH --time=5:00
#SBATCH --chdir=/scratch/crk239/parallel_proj
#SBATCH --mem=10000
#SBATCH -G 1
#SBATCH --constraint=p100
#SBATCH --account=cs450-fall20
#SBATCH --qos=gpu_class

# load a module, for example
module load cuda

# TODO: add actual commands