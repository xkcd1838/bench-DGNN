#!/bin/bash -eu
# Not to be called directly, use nohup_queue.sh instead.
# Expects to be called from one directory up.

ttt=30 #Time to try before terminating in days
sleep_time=1 #in minutes

expfile=$1
expname=${2:-$(basename $expfile)}
tries=$(( ttt*24*60 / sleep_time ))
dir=$(dirname "$0")
c="0"

while [ $c -lt $tries ]; do
  c=$((c+1))
  if [ "$(pgrep -fc 'python run_exp.py --config')" -eq "0" ]; then
    echo "No process, starting up" $expfile
    #Spin up the next thing!
    $dir/nohup_run_gpu.sh $expfile skip_pull $expname
    exit
  else
    echo "Pulse - Process detected, sleeping" $c $expfile
    sleep "$sleep_time"m
  fi
done

