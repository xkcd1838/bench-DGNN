#!/bin/bash -eu
# Expects to be called from one directory up.

mkdir -p out

expfile=$1
skip_pull=${2:-""} # Pass "" if you want to pull but give the experiment a name
expname=${3:-$(basename $expfile)}
dir=$(dirname "$0")
echo $expname

if [ $skip_pull ]; then
  echo "skip pull"
else
  # Ensure we stay in sync
  git pull
fi

i="0"
outfile="out/${expname}.out"
while [ -f "$outfile" ]; do
    echo "$outfile exists."
    i=$[$i+1]
    outfile="out/${expname}${i}.out"
  done
echo "out to $outfile"

# Assuming we don't use the one_cell flag here
nohup $dir/run_gpu.sh $expfile > $outfile  2>&1 &

