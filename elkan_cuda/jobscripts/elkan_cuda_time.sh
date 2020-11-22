#!/bin/bash
#SBATCH --job-name=l_cu_time
#SBATCH --output=/scratch/als872/elkan_cuda_out.txt
#SBATCH --time=5:00
#SBATCH --mem=10000
#SBATCH -G 1
#SBATCH --constraint=p100
#SBATCH --qos=gpu_class

# load a module, for example
module load cuda

nvcc -arch=compute_60 -code=sm_60 -lcuda -Xcompiler -fopenmp elkan_cuda.cu csvparser.c -o elkan_cuda

# run
srun ./elkan_cuda 5 "../data/pairs.csv" "," 0 0
srun ./elkan_cuda 10 "../data/pairs.csv" "," 0 0
srun ./elkan_cuda 15 "../data/pairs.csv" "," 0 0
srun ./elkan_cuda 25 "../data/pairs.csv" "," 0 0