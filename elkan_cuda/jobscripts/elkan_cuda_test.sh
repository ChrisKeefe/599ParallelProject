#!/bin/bash
#SBATCH --job-name=Q3
#SBATCH --output=/scratch/crk239/599ParallelProj/elkan_cuda_out.txt
#SBATCH --time=5:00
#SBATCH --chdir=/scratch/crk239/599ParallelProj/elkan_cuda
#SBATCH --mem=10000
#SBATCH -G 1
#SBATCH --constraint=p100
#SBATCH --account=cs450-fall20
#SBATCH --qos=gpu_class

# load a module, for example
module load cuda

# run
# The final argument triggers run in debug mode with our preselected iris centers
test: lloyds_seq
	./elkan_cuda 3 "../data/iris.csv" "," 1 1 1
