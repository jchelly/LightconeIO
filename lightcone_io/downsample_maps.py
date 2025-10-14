#!/bin/env python

import sys
import os
import shutil
import h5py
import numpy as np
import healpy as hp

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

def shell_filename(basedir, basename, shell_nr, file_nr):
    return  ("%(basedir)s/%(basename)s_shells/shell_%(shell_nr)d/%(basename)s.shell_%(shell_nr)d.%(file_nr)d.hdf5" %
             {"basedir" : basedir, "basename" : basename,
              "shell_nr" : shell_nr, "file_nr" : file_nr})

def get_power(name):
    """
    Return the power parameter for hp.pixelfunc.ud_grade

    This determines if we compute the mean or sum of pixels
    when down-sampling.
    """
    SUM=-2
    MEAN=0
    power = {
        "TotalMass" : SUM,
        "DarkMatterMass" : SUM,
        "NeutrinoMass" : SUM,
        "SmoothedGasMass" : SUM,
        "UnsmoothedGasMass" : SUM,
        "StellarMass" : SUM,
        "BlackHoleMass" : SUM,
        "StarFormationRate" : SUM,
        "XrayErositaLowIntrinsicPhotons" : SUM,
        "XrayErositaLowIntrinsicEnergies" : SUM,
        "XrayErositaHighIntrinsicPhotons" : SUM,
        "XrayErositaHighIntrinsicEnergies" : SUM,
        "XrayROSATIntrinsicPhotons" : SUM,
        "XrayROSATIntrinsicEnergies" : SUM,
        "ComptonY" : MEAN,
        "DopplerB" : MEAN,
        "DM" : MEAN,
    }
    return power[name]

def downsample_maps(input_basedir, output_basedir, basename, shell_nr, new_nside):
    """
    This down samples all of the HEALPix maps in a single file.
    Only works on single file maps.
    """

    # Open the input file
    input_filename = shell_filename(input_basedir, basename, shell_nr, 0)
    print("Opening input: ", input_filename)
    infile = h5py.File(input_filename, "r")
    nr_files = infile["Shell"].attrs["nr_files_per_shell"]
    if nr_files != 1:
        raise Exception("Can only down sample single file outputs!")

    # Create the output file
    output_filename = shell_filename(output_basedir, basename, shell_nr, 0)
    print("Opening output: ", output_filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    outfile = h5py.File(output_filename, "w")

    # Copy metadata
    for group in ("InternalCodeUnits", "Units", "Shell"):
        infile.copy(group, outfile)

    # Loop over maps
    for name in infile:
        input_dataset = infile[name]
        if "nside" in input_dataset.attrs:
            # Read in the full map
            print("  Reading  %s for shell %d of %s" % (name, shell_nr, basename))
            nside = input_dataset.attrs["nside"][0]
            full_map_data = input_dataset[...]
            # Downsample the map
            print("  Sampling %s for shell %d of %s" % (name, shell_nr, basename))
            downsampled_map_data = hp.pixelfunc.ud_grade(full_map_data, new_nside, power=get_power(name))
            # Write the new map to the output file
            print("  Writing  %s for shell %d of %s" % (name, shell_nr, basename))
            output_dataset = outfile.create_dataset(name, data=downsampled_map_data, 
                                                    chunks=input_dataset.chunks,
                                                    compression=input_dataset.compression,
                                                    compression_opts=input_dataset.compression_opts)
            # Copy attributes
            for attr_name, attr_value in input_dataset.attrs.items():
                output_dataset.attrs[attr_name] = attr_value
            output_dataset.attrs["nside"] = (new_nside,)
            output_dataset.attrs["number_of_pixels"] = (len(downsampled_map_data),)

    print("Finished shell %d of %s" % (shell_nr, basename))
    outfile.close()
    infile.close()

if __name__ == "__main__":

    usage = """
    Usage:

    mpirun python3 -m mpi4py ./lightcone_io_downsample_maps.py \\
        input_dir output_dir basename new_nside
"""
    args = {}
    if comm_rank == 0:
        if len(sys.argv) == 5:
            args["input_basedir"]  = sys.argv[1]
            args["output_basedir"] = sys.argv[2]
            args["basename"]       = sys.argv[3]
            args["new_nside"]      = int(sys.argv[4])
        else:
            args = None
    args = comm.bcast(args)
    if args is None:
        if comm_rank == 0:
            print(usage)
        MPI.Finalize()
        sys.exit(0)

    # Determine number of shells
    if comm_rank == 0:
        input_index_filename = "%s/%s_index.hdf5" % (args["input_basedir"], args["basename"])
        with h5py.File(input_index_filename, "r") as infile:
            nr_shells = infile["Lightcone"].attrs["nr_shells"][0]
    else:
        nr_shells = None
    nr_shells = comm.bcast(nr_shells)

    # Parallelise over the shells
    for shell_nr in range(nr_shells):
        if shell_nr % comm_size == comm_rank:
            downsample_maps(args["input_basedir"], args["output_basedir"], args["basename"],
                            shell_nr, args["new_nside"])

    # Copy over the index file
    if comm_rank == 0:
        output_index_filename = "%s/%s_index.hdf5" % (args["output_basedir"], args["basename"])
        shutil.copyfile(input_index_filename, output_index_filename)

    comm.barrier()
    if comm_rank == 0:
        print("All shells done.")
