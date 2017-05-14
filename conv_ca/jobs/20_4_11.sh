#!/bin/bash -l

#PBS -N gridsearch_20_4
#PBS -l walltime=7:30:00
#PBS -l mem=2gb

source activate tf-cpu

cd $PBS_O_WORKDIR
cd ..

python3 conv_ca.py --num_layers=20 --state_size=4 --run=11
