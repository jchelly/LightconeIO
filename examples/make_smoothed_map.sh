#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/smoothed_map.%A.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

mpirun python3 -m mpi4py ./make_smoothed_map.py ${SLURM_ARRAY_TASK_ID}
