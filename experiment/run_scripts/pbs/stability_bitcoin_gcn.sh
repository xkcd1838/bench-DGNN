#!/bin/bash

# Job to submit to a GPU node.

#PBS -l ncpus=24
#PBS -l ngpus=1
#PBS -q gpuq
#PBS -l mem=44GB
#PBS -l walltime=24:00:00 

##PBS -m abe 
##PBS -M your.email@uts.edu.au

cd ${PBS_O_WORKDIR}
./run_scripts/run_gpu.sh config/stability/bitcoin_gcn.yaml --one_cell
