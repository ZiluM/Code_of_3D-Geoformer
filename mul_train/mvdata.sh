#!/bin/bash -l
#PBS -N mv_DaRes
#PBS -A UWAS0131 
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=0:mem=40GB
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -m abe
#PBS -j oe

mvpath="/glade/derecho/scratch/zilumeng/3DGeoData"

cd /glade/work/zilumeng/3D_trans

mv -v /glade/work/zilumeng/3D_trans/Da/res_end $mvpath