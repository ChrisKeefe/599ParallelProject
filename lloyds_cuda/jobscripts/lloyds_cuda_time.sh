#!/bin/bash
#SBATCH --job-name=l_cu_time
#SBATCH --output=/scratch/als872/lloyds_cuda_out.txt
#SBATCH --time=5:00
#SBATCH --mem=10000
#SBATCH -G 1
#SBATCH --constraint=p100
#SBATCH --account=cs450-fall20
#SBATCH --qos=gpu_class

# load a module, for example
module load cuda

# run
./lloyds_cuda 5 "../data/pairs.csv" "," 0 0
./lloyds_cuda 10 "../data/pairs.csv" "," 0 0
./lloyds_cuda 15 "../data/pairs.csv" "," 0 0
./lloyds_cuda 25 "../data/pairs.csv" "," 0 0