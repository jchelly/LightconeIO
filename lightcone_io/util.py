#!/bin/env python

import h5py
import numpy as np


def find_total_nr_particles(basedir, basename):
    """
    Find the total number of particles in a lightcone output.
    The lightcone does not need to be spatially indexed for
    this to work. The last file written by each rank is read
    to get the cumulative particle count for each type.
    """
    
    # Read the number of files written by each rank from the index file
    filename = "%s/%s_index.hdf5" % (basedir, basename)
    with h5py.File(filename, "r") as infile:
        final_particle_file_on_rank = infile["Lightcone"].attrs["final_particle_file_on_rank"]
        nr_mpi_ranks = int(infile["Lightcone"].attrs["nr_mpi_ranks"])

    # Open the last file from each rank    
    counts = {}
    for rank_nr in range(nr_mpi_ranks):
        filename = "%s/%s_particles/%s_%04d.%d.hdf5" % (basedir, basename, basename,
                                                        final_particle_file_on_rank[rank_nr],
                                                        rank_nr)
        with h5py.File(filename, "r") as infile:
            for name, value in infile["Lightcone"].attrs.items():
                if name.startswith("cumulative_count"):
                    if name in counts:
                        counts[name] += value[0]
                    else:
                        counts[name] = value[0]
    return counts
