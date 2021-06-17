#!/bin/bash -e

# Run experiment
singularity exec ../../container.sif python run_exp.py --config_file $1 $2
