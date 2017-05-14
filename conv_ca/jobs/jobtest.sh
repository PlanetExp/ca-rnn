#!/bin/bash -l

#PBS -N gridsearch
#PBS -l walltime=9:00:00
#PBS -l mem=2gb
#PBS -J 2-16

# source activate tf-cpu

# cd $PBS_O_WORKDIR


# params=`sed -n "2 p" input.csv`
# sed -n '2 p' input.csv

IFS=',' read params <<< `sed -n "13 p" input.csv`
paramArray=($params)

i=${paramArray[1]}
j=${paramArray[2]}
k=${paramArray[3]}
echo $i
echo $j
echo $k
# cd ..
# python3 conv_ca.py --num_layers=$i --state_size=$j --run=$k
