#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L1000N1800/hash_%x.lightcone%a.indexed.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#SBATCH --reservation=jlvc76_53
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1
module load python/3.10.1

sim="L1000N1800/${SLURM_JOB_NAME}"
lightcone_nr=${SLURM_ARRAY_TASK_ID}
basename=lightcone${lightcone_nr}
ptype="BH"

# Find all of the particle files
filenames="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcones_z_first_nest/${basename}_particles/*.hdf5"

# Hash the contents
mpirun python3 -u -m mpi4py ../hash_particles.py ${ptype} "${filenames}"
