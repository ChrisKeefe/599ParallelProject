#!/bin/bash
#SBATCH --job-name=seq_and_par
#SBATCH --chdir=/scratch/crk239/599parallel/599ParallelProject/scalability
#SBATCH --nodelist=cn4
#SBATCH --time=60:00
#SBATCH --mem=15000
#SBATCH --cpus-per-task=32

# load a module, for example

# run

./lloyds_par/lloyds_par1 15 "../data/pairs.csv" "," 0 0 >> lloyds_par_out.txt
./lloyds_par/lloyds_par8 15 "../data/pairs.csv" "," 0 0 >> lloyds_par_out.txt
./lloyds_par/lloyds_par16 15 "../data/pairs.csv" "," 0 0 >> lloyds_par_out.txt
./lloyds_par/lloyds_par24 15 "../data/pairs.csv" "," 0 0 >> lloyds_par_out.txt
./lloyds_par/lloyds_par32 15 "../data/pairs.csv" "," 0 0 >> lloyds_par_out.txt

./elkan_par/elkan_par 15 "../data/pairs.csv" "," 0 0 >> elkan_par_out.txt
./elkan_par/elkan_par8 15 "../data/pairs.csv" "," 0 0 >> elkan_par_out.txt
./elkan_par/elkan_par16 15 "../data/pairs.csv" "," 0 0 >> elkan_par_out.txt
./elkan_par/elkan_par24 15 "../data/pairs.csv" "," 0 0 >> elkan_par_out.txt
./elkan_par/elkan_par32 15 "../data/pairs.csv" "," 0 0 >> elkan_par_out.txt
