#PBS -N prime
#PBS -A UBUB0017
#PBS -q main
#PBS -j oe
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=64:ngpus=1
#PBS -l job_priority=regular

module purge
module load conda
module list

conda activate pt212gpu_conda

python3 testrun.py --runname=canon --config=/glade/u/home/cobrien/prime/prime_lib/configs/prime_v2.yaml