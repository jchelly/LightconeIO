#!/bin/env python
#
# This script is used to generate a tiny test SOAP input which can be used
# to make a similarly tiny lightcone halo catalogue.
#
# We random sample the SOAP halos and discard most of their properties. We need
# at least "InputHalos/HaloCentre" and "InputHalos/HaloCatalogueIndex". The
# HBT TrackId is also useful as a cross check.
#
# Need to sample all SOAP outputs for the halo lightcone code to run but we'll
# only keep one (and it's corresponding lightcone slice) in the repository.
#
# The slurm batch script in this directory was used to process the sampled
# SOAP catalogue into a halo lightcone file.
#

import h5py
import numpy as np


def sample_soap(snap_nr):

    print(f"Snapshot {snap_nr:04d}")

    # List of SOAP properties to keep
    to_keep = [
        "InputHalos/HaloCentre",
        "InputHalos/HaloCatalogueIndex",
        "InputHalos/HBTplus/TrackId",
        "SO/200_crit/TotalMass",
    ]

    # Location of the input catalogue
    input_filename = f"/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"

    # Sampling rate, to keep the file small enough to commit to the git repo.
    # This should yield ~2,000 halos.
    sample_rate = 0.0005874 / 5

    # Location of the output
    output_filename = f"/cosma8/data/dp004/jch/halo_properties_{snap_nr:04d}.hdf5"

    # Create the output file
    outfile = h5py.File(output_filename, "w")

    # Open the input file
    infile = h5py.File(input_filename, "r")

    # Copy metadata groups
    for name in ("Cosmology", "Units", "PhysicalConstants"):
        infile.copy(infile[name], outfile, name)

    # Random sample the halos
    rng = np.random.default_rng(seed=0)
    nr_halos = int(infile["Header"].attrs["NumSubhalos_ThisFile"][0])
    keep = rng.random(nr_halos) < sample_rate
    print(f"Number of halos to keep = {sum(keep)}")

    # Copy over the halo data we need
    for name in to_keep:
        print(f"Copying: {name}")
        data = infile[name][...][keep,...]
        outfile.require_group(name.rsplit("/", 1)[0])
        outfile.create_dataset(name, data=data, compression="gzip", compression_opts=9)
        for attr_name, attr_val in infile[name].attrs.items():
            outfile[name].attrs[attr_name] = attr_val

if __name__ == "__main__":

    for snap_nr in range(78):
        sample_soap(snap_nr)
