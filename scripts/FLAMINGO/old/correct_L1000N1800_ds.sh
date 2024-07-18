#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L1000N1800/correct_%x_ds.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

sim="L1000N1800/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}
basename=lightcone${lightcone_nr}
sim_dir=/cosma8/data/dp004/flamingo/Runs/${sim}/
if [[ "$SLURM_JOB_NAME" == DMO_* ]] ; then
  echo Processing DMO run
  input_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/downsampled_maps_4096/
elif [[ "$SLURM_JOB_NAME" == HYDRO_* ]] ; then
  echo Processing hydro run
  input_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/corrected_downsampled_4096/
else
  echo Unrecognised run name
  exit
fi
output_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/neutrino_corrected_maps_downsampled_4096/

\mkdir -p ${output_dir}
lfs setstripe --stripe-count=-1 --stripe-size=32M ${output_dir}

# Assume script is in $PATH
script=`which lightcone_io_correct_maps.py`

# Find simulation config file
yml_file=${sim_dir}/flamingo*.yml

mpirun python3 -m mpi4py ${script} ${input_dir} ${yml_file} ${output_dir} ${sim_dir}/snapshots/flamingo_0000/flamingo_0000.0.hdf5 ${basename}
