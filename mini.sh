#!/bin/bash
#SBATCH --job-name=mini
#SBATCH --chdir=/scratch/crk239/599parallel/599ParallelProject
#SBATCH --time=6:00
#SBATCH --mem=10000

# load a module, for example

# run
./lloyds_seq/lloyds_seq 5 "data/pairs.csv" "," 0 0 >> mini_thing.out