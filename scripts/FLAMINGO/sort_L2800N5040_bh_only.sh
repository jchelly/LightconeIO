#!/bin/bash -l
#
#SBATCH --nodes=4
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L2800N5040/index_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00
#

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# Activate virtual env with lightcone_io installed
source /cosma/apps/dp004/${USER}/lightcone_env/bin/activate

# Job name indicates simulation to process
name=${SLURM_JOB_NAME}

# Job array index indicates lightcone number to do
lightcone_nr=${SLURM_ARRAY_TASK_ID}

# Input lightcone
basedir=/cosma8/data/dp004/flamingo/Runs/L2800N5040/${name}/lightcones/
basename=lightcone${lightcone_nr}

# Number of redshift bins
nr_redshift_bins=64

# HEALPix map resolution and ordering for binning on the sky
nside=16
order="nest"

# Storage parameters
chunksize=1048576

# Output directory
outdir=/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L2800N5040/${name}/black_hole_lightcones/
\mkdir -p ${outdir}
lfs setstripe --stripe-count=4 --stripe-size=32M ${outdir}

# Run the code
mpirun -- python3 -u -m mpi4py -m lightcone_io.index_particles \
       ${basedir} ${basename} ${nr_redshift_bins} ${nside} ${outdir} \
       --order ${order} --redshift-first --lossy --chunksize=${chunksize} \
       --types="BH"
