#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L2800N5040/downsample_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#

new_nside=4096

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

sim="L2800N5040/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}
basename=lightcone${lightcone_nr}

if [[ $SLURM_JOB_NAME == DMO_* ]] ; then
  # DMO maps don't need to be corrected
  input_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/combined_maps/
  output_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcones_downsampled_${new_nside}/
else
  input_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/corrected_maps/
  output_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/corrected_downsampled_${new_nside}/
fi

# Output is a single large file per map, so stripe
\mkdir -p ${output_dir}
lfs setstripe --stripe-count=-1 --stripe-size=32M ${output_dir}

# Assume script is in $PATH
script=`which lightcone_io_downsample_maps.py`

mpirun python3 -m mpi4py ${script} ${input_dir} ${output_dir} ${basename} ${new_nside}
