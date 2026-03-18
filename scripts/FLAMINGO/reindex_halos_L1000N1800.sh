#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L1000N1800/reindex_%x.lightcone%a.%A.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 12:00:00
#

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

#source /cosma/apps/dp004/${USER}/lightcone_env/bin/activate
activate lightcone_io

# Simulation to do (based on job name)
sim="L1000N1800/${SLURM_JOB_NAME}"
first_snap=0
last_snap=77

# Which lightcone to use
lightcone_nr="${SLURM_ARRAY_TASK_ID}"

# Input lightcone halo catalogue
input_halos="/cosma8/data/dp004/flamingo/Runs/${sim}/hbt_lightcone_halos/"

# Set striping on output location
outdir="/cosma8/data/dp004/jch/FLAMINGO/HBT/${sim}/reindexed_halos/lightcone${lighcone_nr}"
\mkdir -p "${outdir}"
lfs setstripe --stripe-count=1 --stripe-size=8M "${outdir}"

# Run the code
mpirun -- python3 -m mpi4py -m lightcone_io.reindex_halos \
       "${first_snap}" "${last_snap}" \
       "${input_halos}" lightcone"${lightcone_nr}" \
       "/cosma8/data/dp004/jch/FLAMINGO/HBT/${sim}/reindexed_halos/" \
       --nside=16 --order=nest \
       --soap-format="/cosma8/data/dp004/flamingo/Runs/${sim}/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"
