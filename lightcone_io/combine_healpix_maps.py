#!/bin/env python

import sys
import os
import h5py
import numpy as np

def map_file_name(basedir, basename, shell_nr, file_nr):
    return ("%s/%s_shells/shell_%d/%s.shell_%d.%d.hdf5" %
            (basedir, basename, shell_nr, basename, shell_nr, file_nr))

def combine_healpix_maps(indir, basename, shell_nr, outdir):

    # Create the output file
    outname = map_file_name(outdir, basename, shell_nr, 0)
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    outfile = h5py.File(outname, "w")
    print("Created output file: %s" % outname)

    # Open the first input file
    inname = map_file_name(indir, basename, shell_nr, 0)
    infile = h5py.File(inname, "r")
    print("Opened input file: %s" % inname)

    # Copy metadata
    infile.copy(source="Units", dest=outfile)
    infile.copy(source="InternalCodeUnits", dest=outfile)
    infile.copy(source="Shell", dest=outfile)
    outfile["Shell"].attrs["nr_files_per_shell"] = 1

    # Get list of datasets
    datasets = []
    for name in infile:
        if "nside" in infile[name].attrs:
            datasets.append(name)
            print("  found dataset %s" % name)

    # Determine total number of pixels
    number_of_pixels = infile[datasets[0]].attrs["number_of_pixels"]
    print("There are %d pixels per map" % number_of_pixels)

    # Create output datasets
    # Note: this will not preserve lossy compression filters
    for name in datasets:
        dset = infile[name]
        shape = (number_of_pixels,)
        outfile.create_dataset(name, shape=shape, dtype=dset.dtype,
                               chunks=dset.chunks, compression=dset.compression,
                               compression_opts=dset.compression_opts,
                               shuffle=dset.shuffle)
        for attr_name in infile[name].attrs:
            outfile[name].attrs[attr_name] = infile[name].attrs[attr_name]
        print("Created output dataset %s" % name)

    # Close the input file
    infile.close()

    # Copy data. Here we need to loop over the input files.
    offset = 0
    file_nr = 0
    while offset < number_of_pixels:
        
        # Open the next input file
        inname = map_file_name(indir, basename, shell_nr, file_nr)
        infile = h5py.File(inname, "r")
        print("Opened input file to copy pixel data: %s" % inname)
        
        # Copy pixel data from this file
        local_pixels = None
        for name in datasets:
            dset = infile[name]

            # Check number of pixels in this file
            if local_pixels is None:
                local_pixels = dset.shape[0]
            else:
                if local_pixels != dset.shape[0]:
                    raise Exception("All maps must be the same size!")
    
            # Copy the pixel data for this map
            outfile[name][offset:offset+local_pixels] = infile[name][:]

        # Next file
        offset += local_pixels
        file_nr += 1

    if offset != number_of_pixels:
        raise Exception("Number of pixels copied is wrong!")

    print("Finished merging files")


if __name__ == "__main__":

    indir    = sys.argv[1]
    basename = sys.argv[2]
    shell_nr = int(sys.argv[3])
    outdir   = sys.argv[4]

    combine_healpix_maps(indir, basename, shell_nr, outdir)

