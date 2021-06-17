#!/bin/bash -eu
# Expects to be called from one directory up.

mkdir -p out

expfile=$1
expname=${2:-$(basename $expfile)}
dir=$(dirname "$0")
echo $expname

i="0"
outfile="out/${expname}.out"
while [ -f "$outfile" ]; do
    echo "$outfile exists."
    i=$[$i+1]
    outfile="out/${expname}${i}.out"
  done
echo "out to $outfile"

# Assuming we don't use the one_cell flag here
nohup $dir/run_cpu.sh $expfile > $outfile  2>&1 &
