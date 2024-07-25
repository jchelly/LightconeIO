#!/bin/bash
#
#SBATCH --nodes=8
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L1000N1800/halo_ids_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 1:30:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

sim="L1000N1800/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}

# Location of the lightcone particle data
lightcone_dir="/cosma8/data/dp004/flamingo/Runs/${sim}/particle_lightcones/"
lightcone_base="lightcone${lightcone_nr}"

# Location of the halo lightcone
halo_lightcone_filenames="/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcone_halos/${lightcone_base}/lightcone_halos_%(file_nr)04d.hdf5"

# Location of SOAP catalogues
soap_filenames="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_%(snap_nr)04d.hdf5"

# Where to write the output
output_dir="/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcone_particle_halo_ids_test/lightcone${lightcone_nr}/"
\mkdir -p ${output_dir}
lfs setstripe --stripe-count=-1 --stripe-size=32M ${output_dir}

# Assume we're running from scripts/FLAMINGO in the source directory
script=../../bin/lightcone_io_particle_halo_ids.py 

mpirun --output-filename halo_ids python3 -m mpi4py ${script} \
    "${lightcone_dir}" \
    "${lightcone_base}" \
    "${halo_lightcone_filenames}" \
    "${soap_filenames}" \
    "${output_dir}" \
    --soap-so-name="SO/200_crit" \
    --overlap-method=fractional-radius

