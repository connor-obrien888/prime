#PBS -N psmap
#PBS -A UBUB0018
#PBS -q main
#PBS -j oe
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=128
#PBS -l job_priority=regular

module purge
module load conda
module list

conda activate pt212gpu_conda

export KAIJU_INSTALL_DIR=/glade/u/home/cobrien/kaiju-private
source $KAIJU_INSTALL_DIR/scripts/setupEnvironment.sh
export KAIJU_BUILD_DIR=/glade/u/home/cobrien/kaiju-private/build_mpi

python3 ps_mapmaker.py