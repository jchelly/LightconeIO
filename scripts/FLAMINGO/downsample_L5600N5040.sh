#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L5600N5040/downsample_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#

new_nside=4096

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

sim="L5600N5040/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}
basename=lightcone${lightcone_nr}

input_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcones/
output_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcones_downsampled/_${nside}

# Assume script is in $PATH
script=`which lightcone_io_downsample_maps.py`

mpirun python3 -m mpi4py ${script} ${input_dir} ${output_dir} ${basename} ${new_nside}
