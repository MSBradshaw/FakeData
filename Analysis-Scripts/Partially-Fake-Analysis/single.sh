#!/bin/bash

#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=8192M   # memory per CPU core
#SBATCH --mail-user=michaelscottbradshaw@gmail.com   # email address
#SBATCH --mail-type=FAIL

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

~/python3/bin/python3 Classification-Downsample-Features.py $1 dumb results.csv PartiallyResampled
