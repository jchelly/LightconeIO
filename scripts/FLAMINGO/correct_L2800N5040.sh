#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L2800N5040/correct_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 2:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

sim="L2800N5040/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}
basename=lightcone${lightcone_nr}

sim_dir=/cosma8/data/dp004/flamingo/Runs/${sim}/
input_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcones/
output_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/corrected_maps/

\mkdir -p ${output_dir}
lfs setstripe --stripe-count=-1 --stripe-size=32M ${output_dir}

# Assume script is in $PATH
script=`which lightcone_io_correct_maps.py`

# Find simulation config file
yml_file=${sim_dir}/flamingo*.yml

mpirun python3 -m mpi4py ${script} ${input_dir} ${yml_file} ${output_dir} ${basename}
