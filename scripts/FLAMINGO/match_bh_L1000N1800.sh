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
tree_basename=/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/${sim}/output/Velociraptor

# Location of the lightcone particle data
lightcone_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcones/
lightcone_base=lightcone${lightcone_nr}

# Where to write the output
output_dir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcone_halos/
\mkdir -p ${output_dir}

# Assume script is in $PATH
script=`which lightcone_io_match_black_holes.py`

mpirun python3 -m mpi4py ${script} ${tree_basename} ${lightcone_dir} ${lightcone_base} ${output_dir}
