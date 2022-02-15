#!/bin/env python

import sys
import glob
import h5py
import numpy as np

def check_index_file(basedir, basename):

    print("Lightcone: ", basename)

    # Read the index
    filename="%s/%s_index.hdf5" % (basedir, basename)
    with h5py.File(filename, "r") as infile:
        nr_mpi_ranks = int(infile["Lightcone"].attrs["nr_mpi_ranks"])
        final_particle_file_on_rank = infile["Lightcone"].attrs["final_particle_file_on_rank"]
        nr_files_per_shell = int(infile["Lightcone"].attrs["nr_files_per_shell"])

    # Count particle files
    print("  Check number of particle files")
    fnames = glob.glob("%s/%s_particles/%s_0000.*.hdf5" % (basedir, basename, basename))
    if len(fnames) == 0:
        # No particle files: check that this is expected from the index file
        if np.all(final_particle_file_on_rank==-1):
            print("    OK: No particle files, as expected")
        else:
            raise Exception("ERROR: Missing particle files!")
    else:
        # Have some particle files. Check expected number
        if(np.all(final_particle_file_on_rank == 0)):
            print("    OK: one file per rank as expected")
        else:
            raise Exception("ERROR: should have one file per rank!")
        # Check number of files
        if len(fnames) == nr_mpi_ranks:
            print("    OK: have expected number of particle files")
        else:
            raise Exception("ERROR: unexpected number of particle files")

    print("  Check lightcone maps")
    if nr_files_per_shell != 1:
        raise Exception("ERROR: should have one file per shell!")
    else:
        print("    OK: one file per shell")


if __name__ == "__main__":

    basedir=sys.argv[1]
    nr_lightcones=int(sys.argv[2])

    for lightcone_nr in range(nr_lightcones):
        check_index_file(basedir, "lightcone%d" % lightcone_nr)
