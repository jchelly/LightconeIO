#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L1000N1800/match_bh_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

sim="L1000N1800/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}

# Location of the merger trees
tree_filename=/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/${sim}/trees_f0.1_min10_max100/vr_trees.hdf5

# Location of the lightcone particle data
lightcone_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/particle_lightcones/
lightcone_base=lightcone${lightcone_nr}

# Where to write the output
output_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcone_halos/lightcone${lightcone_nr}/
\mkdir -p ${output_dir}
lfs setstripe --stripe-count=-1 --stripe-size=32M ${output_dir}

# Assume we're running from scripts/FLAMINGO in the source directory
script=../../bin/lightcone_io_match_black_holes.py 

mpirun python3 -m mpi4py ${script} ${tree_filename} ${lightcone_dir} ${lightcone_base} ${output_dir}
