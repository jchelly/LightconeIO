#!/bin/env python

import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

import lightcone_io.particle_sorter

if __name__ == "__main__":

    usage="""
    Usage: python3 -m mpi4py lightcone_io_index_particles.py \\
              basedir basename nr_redshift_bins nside outdir

    basedir         : lightcone output directory containing index files
    basename        : name of the lightcone to process
    nr_redshift_bins: number of redshift bins to use to bin particles
    nside           : HEALPpix map resolution to use to bin particles
"""

    args = {}
    if comm_rank == 0:
        if len(sys.argv) != 6:
            print(usage)
            args = None
        else:
            args["basedir"]          = sys.argv[1]
            args["basename"]         = sys.argv[2]
            args["nr_redshift_bins"] = int(sys.argv[3])
            args["nside"]            = int(sys.argv[4])
            args["outdir"]           = sys.argv[5]
    args = comm.bcast(args)

    if args is None:
        comm.finalize()
        sys.exit(0)

    lightcone = LightconeSorter(args["basedir"], args["basename"],
                                MPI.COMM_WORLD)
    
    lightcone.write_sorted_lightcone(args["outdir"], args["nr_redshift_bins"],
                                     args["nside"])
