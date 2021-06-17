#!/bin/bash -e

# Run experiment
singularity exec --nv ../../container.sif python run_exp.py --config_file $1 $2
