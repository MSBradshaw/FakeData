#!/bin/bash

#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=8192M   # memory per CPU core
#SBATCH --mail-user=michaelscottbradshaw@gmail.com   # email address
#SBATCH --mail-type=FAIL

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
bash start.sh $1
