#!/bin/bash -eu
# Expects to be called from one directory up.

mkdir -p out/queue

expfile=$1
expname=$(basename $expfile)
dir=$(dirname "$0")
echo $expname

nohup $dir/queue_nohup_gpu_run.sh $expfile > "out/queue/${expname}.out" 2>&1 &
