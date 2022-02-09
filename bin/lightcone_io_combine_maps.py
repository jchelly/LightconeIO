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


def combine_maps_mpi(indir, nr_lightcones, outdir):
    
    if comm_rank == 0:
        # Read the index files to get number of shells
        shells = []
        for lightcone_nr in range(nr_lightcones):
            fname = indir+("/lightcone%d_index.hdf5" % lightcone_nr)
            with h5py.File(fname, "r") as infile:
                nr_shells = infile["Lightcone"].attrs["nr_shells"][0]
            print("Lightcone %d has %d shells" % (lightcone_nr, nr_shells))
            for shell_nr in range(nr_shells):
                shells.append((lightcone_nr, shell_nr))
    else:
        shells = None
    shells = comm.bcast(shells)

    # Merge the maps
    for index, (lightcone_nr, shell_nr) in enumerate(shells):
        basename = "lightcone%d" % lightcone_nr
        if index % comm_size == comm_rank:
            hm.combine_healpix_maps(indir, basename, shell_nr, outdir)

    # Copy and update the index files if necessary
    if comm_rank == 0:
        for lightcone_nr in range(nr_lightcones):
            input_index  = indir+("/lightcone%d_index.hdf5" % lightcone_nr)
            output_index = outdir+("/lightcone%d_index.hdf5" % lightcone_nr)
            if not(os.path.exists(output_index)):
                shutil.copyfile(input_index, output_index)
            with h5py.File(output_index, "r+") as indexfile:
                indexfile["Lightcone"].attrs["nr_files_per_shell"] = [1,]


if __name__ == "__main__":
    
    usage="""
    Usage: python3 -m mpi4py lightcone_io_combine_maps.py \\
              basedir nr_lightcones outdir

    basedir      : location of the input lightcones
    nr_lightcones: number of lightcones to process
    outdir       : where to write the output
"""

    args = {}
    if comm_rank == 0:
        if len(sys.argv) != 4:
            print(usage)
            args = None
        else:
            args["indir"] = sys.argv[1]
            args["nr_lightcones"] = int(sys.argv[2])
            args["outdir"] = sys.argv[3]
    args = comm.bcast(args)

    if args is None:
        MPI.Finalize()
        sys.exit(0)

    combine_maps_mpi(args["indir"], args["nr_lightcones"], args["outdir"])

    comm.barrier()
    if comm_rank == 0:
        print("All shells done.")
