#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=70
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L1000N1800/%x.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 12:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

sim="L1000N1800/${SLURM_JOB_NAME}"

input_dir=/cosma8/data/dp004/jlvc76/FLAMINGO/ScienceRuns/${sim}/lightcones/
output_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcones/

nr_lightcones=2

# Assume script is in $PATH
script=`which lightcone_io_combine_maps.py`

mpirun python3 -m mpi4py ${script} \
    ${input_dir} ${nr_lightcones} ${output_dir}  
