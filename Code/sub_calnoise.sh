#!/bin/bash -l
#PBS -N e2mip_job
#PBS -A UWAS0131 
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1:mem=30GB
#PBS -l gpu_type=v100
#PBS -l walltime=6:00:00
#PBS -q casper
#PBS -m abe
#PBS -j oe


module load conda
conda activate earth2mip

 

### Debugging

nvidia-smi
# module list
# python -c "import torch; print(torch.cuda.is_available())"


### Run the model

cd /glade/work/zilumeng/3D_trans

python ./Code/save_error.py