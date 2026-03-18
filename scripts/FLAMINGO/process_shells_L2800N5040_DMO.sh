#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L2800N5040/process_maps_%x.%a.out
#SBATCH -p cosma8
#SBATCH -A dp203
#SBATCH -t 7:00:00
#

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# Activate virtual env with lightcone_io installed
source /cosma/apps/dp004/${USER}/lightcone_env/bin/activate

sim="L2800N5040/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}
basename=lightcone${lightcone_nr}

# #
# # Combine original outputs to single files
# #

# input_dir=/cosma8/data/dp004/flamingo/Runs/${sim}/lightcones-do-not-use/
# output_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/combined_maps/

# # Output is a single large file per map, so stripe
# \mkdir -p ${output_dir}
# lfs setstripe --stripe-count=4 --stripe-size=32M ${output_dir}

# mpirun -- python3 -m mpi4py -m lightcone_io.combine_maps \
#     ${input_dir} ${output_dir} ${basename}

#
# Apply corrections
#

sim_dir=/cosma8/data/dp004/flamingo/Runs/${sim}/
input_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/combined_maps/
output_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/neutrino_corrected_maps/

\mkdir -p ${output_dir}
lfs setstripe --stripe-count=4 --stripe-size=32M ${output_dir}

# Find simulation config file
if [ -f "${sim_dir}/params.yml" ] ; then
    # DCDM runs have a different name
    yml_file="${sim_dir}/params.yml"
else
    yml_file="${sim_dir}"/flamingo*.yml
fi

mpirun -- python3 -m mpi4py -m lightcone_io.correct_maps \
       ${input_dir} ${yml_file} ${output_dir} ${sim_dir}/snapshots/flamingo_0078/flamingo_0078.0.hdf5 ${basename}

# #
# # Downsample. Uses a lot of memory per process, so limit processes.
# #
# new_nside=4096
# input_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/neutrino_corrected_maps/
# output_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/neutrino_corrected_maps_downsampled_${new_nside}/

# # Output is a single large file per map, so stripe
# \mkdir -p ${output_dir}
# lfs setstripe --stripe-count=4 --stripe-size=32M ${output_dir}

# mpirun -np 4 -- python3 -m mpi4py -m lightcone_io.downsample_maps \
#        ${input_dir} ${output_dir} ${basename} ${new_nside}
