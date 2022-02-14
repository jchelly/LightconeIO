#!/bin/env python3

import sys
import os.path
import shutil

import h5py

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import lightcone_io.healpix_maps as hm


def combine_maps_mpi(indir, outdir, basenames):
    
    if comm_rank == 0:
        # Read the index files to get number of shells
        shells = []
        for basename in basenames:
            fname = indir+("/%s_index.hdf5" % basename)
            with h5py.File(fname, "r") as infile:
                nr_shells = infile["Lightcone"].attrs["nr_shells"][0]
            print("Lightcone %s has %d shells" % (basename, nr_shells))
            for shell_nr in range(nr_shells):
                shells.append((basename, shell_nr))
    else:
        shells = None
    shells = comm.bcast(shells)

    # Merge the maps
    for index, (basename, shell_nr) in enumerate(shells):
        if index % comm_size == comm_rank:
            hm.combine_healpix_maps(indir, basename, shell_nr, outdir)

    # Copy and update the index files if necessary
    if comm_rank == 0:
        for basename in basenames:
            input_index  = indir+("/%s_index.hdf5" % basename)
            output_index = outdir+("/%s_index.hdf5" % basename)
            if not(os.path.exists(output_index)):
                shutil.copyfile(input_index, output_index)
            with h5py.File(output_index, "r+") as indexfile:
                indexfile["Lightcone"].attrs["nr_files_per_shell"] = [1,]


if __name__ == "__main__":
    
    usage="""
    Usage: python3 -m mpi4py lightcone_io_combine_maps.py \\
              basedir outdir basename1 basename2 ...

    basedir      : location of the input lightcones
    outdir       : where to write the output
    basename*    : basenames of lightcones to process
"""

    args = {}
    if comm_rank == 0:
        if len(sys.argv) < 4:
            print(usage)
            args = None
        else:
            args["indir"] = sys.argv[1]
            args["outdir"] = sys.argv[2]
            args["basenames"] = sys.argv[3:]
    args = comm.bcast(args)

    if args is None:
        MPI.Finalize()
        sys.exit(0)

    combine_maps_mpi(args["indir"], args["outdir"], args["basenames"])

    comm.barrier()
    if comm_rank == 0:
        print("All shells done for lightcones ", args["basenames"])
