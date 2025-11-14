#PBS -N indexer
#PBS -A UBUB0017
#PBS -q main
#PBS -j oe
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=128
#PBS -l job_priority=regular

module purge
module load conda
module list

conda activate pt212gpu_conda

python3 indexscript.py