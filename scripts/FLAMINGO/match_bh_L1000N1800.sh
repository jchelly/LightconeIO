#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./test-logs/L1000N1800/match_bh_%x.lightcone%a.%A.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 4:00:00
#

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

#source /cosma/apps/dp004/${USER}/lightcone_env/bin/activate
activate lightcone_io

# Simulation to do (based on job name)
sim="L1000N1800/${SLURM_JOB_NAME}"

# Which lightcone to use
lightcone_nr="${SLURM_ARRAY_TASK_ID}"

# Location of the halo catalogues
halo_type="HBTplus"
halo_format="/cosma8/data/dp004/flamingo/Runs/${sim}/HBT/{snap_nr:03d}/SubSnap_{snap_nr:03d}.{file_nr}.hdf5"

# Lightcone output dir
lightcone_dir="/cosma8/data/dp004/flamingo/Runs/${sim}/bh_particle_lightcones/"

# Location of snapshot files
snapshot_format="/cosma8/data/dp004/flamingo/Runs/${sim}/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5"

# Location of membership files
membership_format="/cosma8/data/dp004/flamingo/Runs/${sim}/SOAP-HBT/membership_{snap_nr:04d}/membership_{snap_nr:04d}.{file_nr}.hdf5"

# Where to write the output
output_dir=/snap8/scratch/dp004/${USER}/FLAMINGO/HBT/${sim}/lightcone_halos/lightcone${lightcone_nr}/
\mkdir -p ${output_dir}
lfs setstripe --stripe-count=1 --stripe-size=8M ${output_dir}

# Run
mpirun -- python3 -m mpi4py -m lightcone_io.match_black_holes \
       "${halo_format}" 0 77 70 70 "${lightcone_dir}" "lightcone${lightcone_nr}" "${snapshot_format}" "${membership_format}" "${output_dir}" \
       --halo-type=HBTplus \
       --pass-through="InputHalos/IsCentral,InputHalos/NumberOfBoundParticles,BoundSubhalo/TotalMass,InputHalos/HBTplus/TrackId"
