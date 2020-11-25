#!/bin/bash
#SBATCH --job-name=dummy
#SBATCH --time=5:00
#SBATCH --chdir=/scratch/crk239/599parallel/599ParallelProject/lloyds_par
#SBATCH --mem=10000
#SBATCH --account=cs450-fall20
#SBATCH --qos=gpu_class

# load a module, for example

# run
bash ./lloyds_par 5 "data/pairs.csv" "," 0 0
