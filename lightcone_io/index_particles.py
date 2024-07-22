#!/bin/env python3

import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

from lightcone_io.particle_sorter import LightconeSorter


if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(comm=comm, description="Reorder SWIFT particle lightcone outputs.")
    parser.add_argument("basedir", type=str, help="Lightcone output directory containing index files")
    parser.add_argument("basename", type=str, help="Name of the lightcone to process (e.g. lightcone0)")
    parser.add_argument("nr_redshift_bins", type=int, help="Number of redshift bins to use to bin particles")
    parser.add_argument("nside", type=int, help="HEALPpix map resolution to use to bin particles")
    parser.add_argument("outdir", type=str, help="Directory in which to write the output")
    parser.add_argument("--order", choices=["nest","ring"], default="nest", help="HEALPix pixel ordering scheme")
    parser.add_argument("--redshift-first", action="store_true", help="Sort by redshift then pixel (True) or pixel then redshift (False)")
    parser.add_argument("--lossy", action="store_true", help="Apply lossy compression to the output")
    parser.add_argument("--chunksize", type=int, default=0, help="HDF5 chunk size for output datasets")
    parser.add_argument("--types", type=str, default=None, help="Comma separated list of particle types to process (BH/DM/Gas/Stars/Neutrino)")
    args = parser.parse_args()

    # Seperate particle types list
    if args.types is not None:
        types = [t.strip() for t in args.types.split(",")]
    else:
        types = None
    
    if comm_rank == 0:
        if types is None:
            print("Processing all particle types")
        else:
            print("Types to process = ", types)
        print("Reading metadata")
    lightcone = LightconeSorter(args.basedir, args.basename, comm, types=types)

    if comm_rank == 0:
        print("Sorting lightcone")    
    lightcone.write_sorted_lightcone(args.outdir, args.nr_redshift_bins,
                                     args.nside, args.order, args.redshift_first,
                                     args.lossy, args.chunksize)
    comm.barrier()
    if comm_rank == 0:
        print("Done.")
