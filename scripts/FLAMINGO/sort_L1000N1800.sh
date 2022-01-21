#!/bin/bash -l
#
#SBATCH --nodes=3
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L1000N1800/index_%x.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#

module purge
module load python/3.10.1
module load gnu_comp/11.1.0
module load openmpi/4.1.1

name=${SLURM_JOB_NAME}

# Input lightcone
basedir=/cosma8/data/dp004/jlvc76/FLAMINGO/ScienceRuns/L1000N1800/${name}/lightcones/
basename=lightcone0

# Binning
nr_redshift_bins=4
nside=32

# Output directory
outdir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/${name}/lightcones/

# Assume script is in $PATH
script=`which lightcone_io_index_particles.py`

mpirun python3 -u -m mpi4py ./${script} \
    ${basedir} ${basename} ${nr_redshift_bins} ${nside} ${outdir}
