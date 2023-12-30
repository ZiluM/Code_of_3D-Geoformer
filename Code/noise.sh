#!/bin/bash -l
#PBS -N e2mip_job
#PBS -A UWAS0131 
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=0:mem=60GB
#PBS -l walltime=6:00:00
#PBS -q casper
#PBS -m abe
#PBS -j oe


module load conda
conda activate earth2mip

python /glade/work/zilumeng/3D_trans/Code/del_noise.py