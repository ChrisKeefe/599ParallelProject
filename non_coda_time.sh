#!/bin/bash
#SBATCH --job-name=seq_and_par
#SBATCH --chdir=/scratch/crk239/599ParallelProj
#SBATCH --time=60:00
#SBATCH --mem=15000
#SBATCH --cpus-per-task=32

# load a module, for example

# run
./lloyds_seq/lloyds_seq 5 "data/pairs.csv" "," 0 0 >> elkan_seq_out.txt
./lloyds_seq/lloyds_seq 10 "data/pairs.csv" "," 0 0 >> elkan_seq_out.txt
./lloyds_seq/lloyds_seq 15 "data/pairs.csv" "," 0 0 >> elkan_seq_out.txt
./lloyds_seq/lloyds_seq 25 "data/pairs.csv" "," 0 0 >> elkan_seq_out.txt

./lloyds_par/lloyds_par 5 "data/pairs.csv" "," 0 0 >> elkan_par_out.txt
./lloyds_par/lloyds_par 10 "data/pairs.csv" "," 0 0 >> elkan_par_out.txt
./lloyds_par/lloyds_par 15 "data/pairs.csv" "," 0 0 >> elkan_par_out.txt
./lloyds_par/lloyds_par 25 "data/pairs.csv" "," 0 0 >> elkan_par_out.txt

./elkan_seq/elkan_seq 5 "data/pairs.csv" "," 0 0 >> elkan_seq_out.txt
./elkan_seq/elkan_seq 10 "data/pairs.csv" "," 0 0 >> elkan_seq_out.txt
./elkan_seq/elkan_seq 15 "data/pairs.csv" "," 0 0 >> elkan_seq_out.txt
./elkan_seq/elkan_seq 25 "data/pairs.csv" "," 0 0 >> elkan_seq_out.txt

./elkan_par/elkan_par 5 "data/pairs.csv" "," 0 0 >> elkan_par_out.txt
./elkan_par/elkan_par 10 "data/pairs.csv" "," 0 0 >> elkan_par_out.txt
./elkan_par/elkan_par 15 "data/pairs.csv" "," 0 0 >> elkan_par_out.txt
./elkan_par/elkan_par 25 "data/pairs.csv" "," 0 0 >> elkan_par_out.txt
