#!/bin/bash -l

module purge
activate lightcone_io
module load gnu_comp/14.1.0 openmpi/5.0.3
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

name="HYDRO_FIDUCIAL"
lightcone_nr=0

# Input lightcone
basedir=/cosma8/data/dp004/jch/test_particles/
basename=lightcone${lightcone_nr}

# Number of redshift bins
nr_redshift_bins=8

# HEALPix map resolution and ordering for binning on the sky
nside=2
order="nest"

# Storage parameters
lossy=1
chunksize=1024

# Output directory
outdir=/cosma8/data/dp004/jch/sorted_test_particles/
\mkdir -p ${outdir}
lfs setstripe --stripe-count=1 --stripe-size=32M ${outdir}

# Run the code
mpirun -np 4 -- python3 -u -m mpi4py -m lightcone_io.index_particles \
       ${basedir} ${basename} ${nr_redshift_bins} ${nside} ${outdir} \
       --order ${order} --redshift-first --chunksize=128 \
       --types="BH"
