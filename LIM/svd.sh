#!/bin/bash -l
#PBS -N jas
#PBS -A UWAS0131 
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=60GB
#PBS -l walltime=02:20:00
#PBS -q casper
#PBS -m abe
#PBS -j oe

module load conda
conda activate earth2mip

cd /glade/work/zilumeng/3D_trans

python3 ./LIM/eof_sst.py