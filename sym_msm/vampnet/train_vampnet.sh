#!/bin/bash

# Submit to the tcb partition
#SBATCH -p lindahl2,lindahl3,lindahl4

# The name of the job in the queue
#SBATCH -J train_vampnet
# wall-clock time given to this job
#SBATCH -t 24:00:00

# Number of nodes and number of MPI processes per node
#SBATCH -N 1 
# Request a GPU node and two GPUs (per node)
#SBATCH -C gpu 
#SBATCH -G 2
#SBATCH --mem=50G


# Output file names for stdout and stderr
#SBATCH -e jupyter.err -o jupyter.out

# Receive e-mails when your job starts and ends

#SBATCH -d singleton

# The actual script starts here

/nethome/yzhuang/anaconda3/envs/deeplearning/bin/python train_vampnet.py
