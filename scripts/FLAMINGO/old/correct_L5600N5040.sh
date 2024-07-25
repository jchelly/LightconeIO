#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L5600N5040/correct_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

sim="L5600N5040/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}
basename=lightcone${lightcone_nr}

sim_dir=/cosma8/data/dp004/flamingo/Runs/${sim}/
input_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/combined_maps/
output_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/neutrino_corrected_maps/

\mkdir -p ${output_dir}
lfs setstripe --stripe-count=-1 --stripe-size=32M ${output_dir}

# Assume script is in $PATH
script=`which lightcone_io_correct_maps.py`

# Find simulation config file
yml_file=${sim_dir}/flamingo*.yml

mpirun python3 -m mpi4py ${script} ${input_dir} ${yml_file} ${output_dir} ${sim_dir}/snapshots/flamingo_0000/flamingo_0000.0.hdf5 ${basename}
