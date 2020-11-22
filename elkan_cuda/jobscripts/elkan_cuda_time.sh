#!/bin/bash
#SBATCH --job-name=l_cu_time
#SBATCH --output=/scratch/als872/ekan_cuda_out.txt
#SBATCH --time=5:00
#SBATCH --mem=10000
#SBATCH -G 1
#SBATCH --constraint=p100
#SBATCH --qos=gpu_class

# load a module, for example
module load cuda

nvcc -arch=compute_60 -code=sm_60 -lcuda -Xcompiler -fopenmp ekan_cuda.cu csvparser.c -o ekan_cuda

# run
srun ./ekan_cuda 5 "../data/pairs.csv" "," 0 0
srun ./ekan_cuda 10 "../data/pairs.csv" "," 0 0
srun ./ekan_cuda 15 "../data/pairs.csv" "," 0 0
srun ./ekan_cuda 25 "../data/pairs.csv" "," 0 0