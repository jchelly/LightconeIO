#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L1000N0900/choose_bh_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 3:00:00
#

module purge
module load gnu_comp/14.1.0 openmpi
module load python/3.12.4

sim="L1000N0900/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}

# Location of the halo catalogues
halo_format="/cosma8/data/dp004/jch/FLAMINGO/HBT/${sim}/SOAP_uncompressed/HBTplus/halo_properties_{snap_nr:04d}.hdf5"

# Location of snapshot files
snapshot_format="/cosma8/data/dp004/flamingo/Runs/${sim}/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5"

# Location of membership files
membership_format="/cosma8/data/dp004/jch/FLAMINGO/HBT/${sim}/SOAP_uncompressed/HBTplus/membership_{snap_nr:04d}/membership_{snap_nr:04d}.{file_nr}.hdf5"

# Where to write the output
output_dir=/snap8/scratch/dp004/jch/FLAMINGO/Tests/${sim}/lightcone_halos/lightcone${lightcone_nr}/
\mkdir -p ${output_dir}
lfs setstripe --stripe-count=-1 --stripe-size=32M ${output_dir}

# Snapshot to do
snap_nr=76

mpirun python3 -m mpi4py -m lightcone_io.choose_bh_tracer \
       "${halo_format}" "${snapshot_format}" "${membership_format}" 75 77 "${snap_nr}" "${output_dir}/test.hdf5"
