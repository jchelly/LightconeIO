#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L1000N1800/combine_maps_%x.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 48:00:00
#

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# Activate virtual env with lightcone_io installed
source /cosma/apps/dp004/${USER}/lightcone_env/bin/activate

sim="L1000N1800/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}
basename=lightcone${lightcone_nr}

input_dir=/cosma8/data/dp004/flamingo/Runs/${sim}/lightcones/
output_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/combined_maps/

# Output is a single large file per map, so stripe
\mkdir -p ${output_dir}
lfs setstripe --stripe-count=-1 --stripe-size=32M ${output_dir}

mpirun -- python3 -m mpi4py -m lightcone_io.combine_maps \
    ${input_dir} ${output_dir} ${basename}
