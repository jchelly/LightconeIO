#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=70
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L5600N5040/combine_maps.%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

sim="L5600N5040/${SLURM_JOB_NAME}"
basename=lightcone${SLURM_ARRAY_TASK_ID}

input_dir=/cosma8/data/dp004/flamingo/Runs/${sim}/lightcones/
output_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/combined_maps/

# Output is a single large file per map, so stripe
\mkdir -p ${output_dir}
lfs setstripe --stripe-count=-1 --stripe-size=32M ${output_dir}

# Assume script is in $PATH
script=`which lightcone_io_combine_maps.py`

mpirun python3 -m mpi4py ${script} \
    ${input_dir} ${output_dir} ${basename}
