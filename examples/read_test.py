#!/bin/env python

import time
import random
import numpy as np
import h5py

#basedir="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcones/"
basedir="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcones_z_first_nest/"
basename="lightcone1"
nr_files = 128

for read_size in (10**4, 10**5, 10**6, 5*10**6, 10**7, 5*10**7):

    # Pick a random file
    file_nr = random.randrange(nr_files)
    filename = "%s/%s_particles/%s_%04d.%d.hdf5" % (basedir, basename, basename, 0, file_nr)
    f = h5py.File(filename, "r")

    # Open the dataset
    dset = f["BH/Coordinates"]

    # Start the clock
    t0 = time.time()

    # Loop over blocks to read
    nr_blocks = 10
    bytes_read = 0
    for block_nr in range(nr_blocks):
        offset = random.randrange(dset.shape[0]-read_size)
        if offset < 0:
            raise Exception("negative offset")
        data = dset[offset:offset+read_size,:]
        bytes_read += (data.dtype.itemsize*3*data.shape[0])

    # Stop the clock
    t1 = time.time()

    # Report speed
    rate = bytes_read/(t1-t0)/(1024**2)
    print("File %5d, block size %12d, read rate = %.2f Mb/s" % (file_nr, read_size, rate))

f.close()
