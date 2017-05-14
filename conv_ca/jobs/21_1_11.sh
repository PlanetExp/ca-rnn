#!/bin/bash -l

#PBS -N gridsearch_21_1
#PBS -l walltime=6:00:00
#PBS -l mem=2gb

source activate tf-cpu

cd $PBS_O_WORKDIR

python3 ../conv_ca.py --num_layers=21 --state_size=1 --run=11
