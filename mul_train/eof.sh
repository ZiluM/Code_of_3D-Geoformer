#!/bin/bash -l
#PBS -N noise_eof1
#PBS -A UWAS0131 
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=0:mem=100GB
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -m abe
#PBS -j oe


module load conda
conda activate earth2mip

cd /glade/work/zilumeng/3D_trans

python3 ./mul_train/noise_eof1.py

