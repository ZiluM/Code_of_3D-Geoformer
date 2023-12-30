#!/bin/bash -l
#PBS -N enso_da_dl_01
#PBS -A UWAS0131 
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1:mem=40GB
#PBS -l gpu_type=v100 
#PBS -l walltime=03:00:00
#PBS -q casper
#PBS -m abe
#PBS -j oe

module load conda
conda activate earth2mip

# execcasper --ngpus 1 --gpu v100 -l walltime=01:00:00 -A UWAS0131 --mem 20GB
# 

### Debugging

nvidia-smi
# module list
# python -c "import torch; print(torch.cuda.is_available())"


### Run the model

cd /glade/work/zilumeng/3D_trans

python /glade/work/zilumeng/3D_trans/Da/da.py /glade/work/zilumeng/3D_trans/Da/config_moreobs_mean3_dl.yml 

# echo "Done, sub post da"

