#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L1000N1800/test_order_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 2:00:00
#

module purge
module load python/3.10.1
module load gnu_comp/11.1.0
module load openmpi/4.1.1

export OPENBLAS_NUM_THREADS=1

name=${SLURM_JOB_NAME}
lightcone_nr=${SLURM_ARRAY_TASK_ID}

# Input lightcone
basedir=/cosma8/data/dp004/flamingo/Runs/L1000N1800/${name}/lightcones/
basename=lightcone${lightcone_nr}

# Number of redshift bins
nr_redshift_bins=16

# HEALPix map resolution and ordering for binning on the sky
nside=16
order="nest"

# Whether to sort by redshift first then pixel (1) or pixel first then redshift (0)
redshift_first=1

# Output directory
outdir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/${name}/lightcones_z_first_nest

# Assume script is in $PATH
script=`which lightcone_io_index_particles.py`

mpirun python3 -u -m mpi4py ${script} \
    ${basedir} ${basename} ${nr_redshift_bins} ${nside} ${order} ${redshift_first} ${outdir}
