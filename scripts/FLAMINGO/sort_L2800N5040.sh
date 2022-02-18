#!/bin/bash -l
#
#SBATCH --nodes=60
#SBATCH --tasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L2800N5040/index_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#SBATCH --reservation=jlvc76_53

module purge
module load python/3.10.1
module load gnu_comp/11.1.0
module load openmpi/4.1.1

export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

name=${SLURM_JOB_NAME}
lightcone_nr=${SLURM_ARRAY_TASK_ID}

# Input lightcone
basedir=/cosma8/data/dp004/flamingo/Runs/L2800N5040/${name}/lightcones/
basename=lightcone${lightcone_nr}

# Number of redshift bins
nr_redshift_bins=64

# HEALPix map resolution and ordering for binning on the sky
nside=16
order="nest"

# Whether to sort by redshift first then pixel (1) or pixel first then redshift (0)
redshift_first=1

# Output directory
outdir=/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/L2800N5040/${name}/indexed_lightcones/

# Assume script is in $PATH
script=`which lightcone_io_index_particles.py`

mpirun python3 -u -m mpi4py ${script} \
    ${basedir} ${basename} ${nr_redshift_bins} ${nside} ${order} ${redshift_first} ${outdir}
