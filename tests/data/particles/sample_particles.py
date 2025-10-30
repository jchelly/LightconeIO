#!/bin/env python

import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

particle_filename = "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/bh_particle_lightcones/lightcone0_particles/lightcone0_0000.{file_nr}.hdf5"

# Sampling rate to get ~10,000 particles
sample_rate = 7.8125e-07

# Quantities to keep (we discard most to keep the file size down)
names = ("Coordinates", "DynamicalMasses", "ExpansionFactors", "ParticleIDs")

# Where to write the output
output_filename = "/cosma8/data/dp004/jch/test_particles/lightcone0_0000.{file_nr}.hdf5"


def sample_file(file_nr):
    print(f"File {file_nr}")
    rng = np.random.default_rng(seed=file_nr)
    with h5py.File(particle_filename.format(file_nr=file_nr), "r") as infile:
        with h5py.File(output_filename.format(file_nr=file_nr), "w") as outfile:
            infile.copy("InternalCodeUnits", outfile)
            infile.copy("Units", outfile)
            infile.copy("Lightcone", outfile)
            outfile.create_group("BH")
            keep = None
            for name in names:
                print(f"  Dataset: {name}")
                data = infile["BH"][name][...]
                if keep is None:
                    keep = (rng.random(data.shape[0]) < sample_rate)
                data = data[keep,...]
                dset = outfile["BH"].create_dataset(name, data=data) #, compression="gzip", compression_opts=9)
                for attr_name, attr_val in infile["BH"][name].attrs.items():
                    dset.attrs[attr_name] = attr_val

def sample_particles():

    # Determine how many files we have
    with h5py.File(particle_filename.format(file_nr=0), "r") as infile:
        nr_files = infile["Lightcone"].attrs["nr_mpi_ranks"][0]

    # Run in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        result = list(executor.map(sample_file, range(nr_files)))


if __name__ == "__main__":
    sample_particles()
