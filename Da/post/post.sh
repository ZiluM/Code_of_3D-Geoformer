#!/bin/bash -l
#PBS -N DL_3_0.7
#PBS -A UWAS0131 
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=0:mem=40GB
#PBS -l walltime=01:30:00
#PBS -q casper
#PBS -m abe
#PBS -j oe

module load conda
conda activate earth2mip

# execcasper --ngpus 1 --gpu v100 -l walltime=01:00:00 -A UWAS0131 --mem 20GB
 

### Debugging

# nvidia-smi
# module list
# python -c "import torch; print(torch.cuda.is_available())"


### Run the model

cd /glade/work/zilumeng/3D_trans

python /glade/work/zilumeng/3D_trans/Da/post/post.py /glade/work/zilumeng/3D_trans/Da/cfg_new3.yml