#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L1000N1800/match_bh_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 3:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

sim="L1000N1800/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}

# Location of the merger trees
tree_filename=/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/${sim}/trees_f0.1_min10_max100_vpeak/vr_trees.hdf5

# Location of the lightcone particle data
lightcone_dir=/cosma8/data/dp004/flamingo/Runs/${sim}/particle_lightcones/
lightcone_base=lightcone${lightcone_nr}

# Location of snapshot files
snapshot_format="/cosma8/data/dp004/flamingo/Runs/${sim}/snapshots/flamingo_%(snap_nr)04d/flamingo_%(snap_nr)04d"

# Determine name of halo membership directories
if [ -e /cosma8/data/dp004/flamingo/Runs/${sim}/SOAP/membership_0077/membership_0077.0.hdf5 ] ; then
  membership_dir=membership
  membership_name=membership
else
  membership_dir=membership
  membership_name=vr_membership
fi

# Location of halo membership files
membership_format="/cosma8/data/dp004/flamingo/Runs/${sim}/SOAP/${membership_dir}_%(snap_nr)04d/${membership_name}_%(snap_nr)04d"
echo Membership files are at ${membership_format}

# Where to write the output
#output_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcone_halos/lightcone${lightcone_nr}/
output_dir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcone_halos/lightcone${lightcone_nr}/
\mkdir -p ${output_dir}
lfs setstripe --stripe-count=-1 --stripe-size=32M ${output_dir}

# Assume we're running from scripts/FLAMINGO in the source directory
script=../../bin/lightcone_io_match_black_holes.py 

mpirun -tag-output python3 -m mpi4py ${script} ${tree_filename} ${lightcone_dir} ${lightcone_base} ${snapshot_format} ${membership_format} ${output_dir}
