#!/bin/env python3

import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

from lightcone_io.particle_sorter import LightconeSorter


def main():

    usage="""
    Usage: python3 -m mpi4py lightcone_io_index_particles.py \\
              basedir basename nr_redshift_bins nside outdir

    basedir         : lightcone output directory containing index files
    basename        : name of the lightcone to process
    nr_redshift_bins: number of redshift bins to use to bin particles
    nside           : HEALPpix map resolution to use to bin particles
    lossy           : (optional) 1=use lossy compression, 0=lossless only

    If a dataset in the input uses lossy compression and lossy=1, then
    that dataset will use lossy compression in the output. Note that this
    will change the values. If lossy=0, then lossy compression is not used
    for any dataset.
"""

    args = {}
    if comm_rank == 0:
        if len(sys.argv) < 8 or len(sys.argv) > 10:
            print(usage)
            args = None
        else:
            args["basedir"]          = sys.argv[1]
            args["basename"]         = sys.argv[2]
            args["nr_redshift_bins"] = int(sys.argv[3])
            args["nside"]            = int(sys.argv[4])
            args["order"]            = sys.argv[5]
            args["redshift_first"]   = bool(int(sys.argv[6]))
            args["outdir"]           = sys.argv[7]
            if len(sys.argv) >= 9:
                args["lossy"] = bool(int(sys.argv[8]))
            else:
                args["lossy"] = True
            if len(sys.argv) == 10:
                args["chunksize"] = int(sys.argv[9])
            else:
                args["chunksize"] = 0

    args = comm.bcast(args)
    if args is None:
        comm.Abort()

    if comm_rank == 0:
        print("Reading metadata")
    lightcone = LightconeSorter(args["basedir"], args["basename"],
                                MPI.COMM_WORLD)

    if comm_rank == 0:
        print("Sorting lightcone")    
    lightcone.write_sorted_lightcone(args["outdir"], args["nr_redshift_bins"],
                                     args["nside"], args["order"], args["redshift_first"],
                                     args["lossy"], args["chunksize"])

    comm.barrier()
    if comm_rank == 0:
        print("Done.")
