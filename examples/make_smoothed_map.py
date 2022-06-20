#!/bin/env python

import sys

import h5py
import numpy as np
import lightcone_io.smoothed_map as smoothed_map
import virgo.mpi.parallel_hdf5 as phdf5

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()


def L1000N1800_HYDRO_FIDUCIAL_smoothed_gas_mass(shell_nr):

    nside = 16384
    input_filename = "/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"
    output_filename = f"/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/new_maps/smoothed_gas_map.shell_{shell_nr}.hdf5"
    shell_redshifts = "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/shell_redshifts_z3.txt"

    # Read shell redshifts
    if comm_rank == 0:
        redshifts = np.loadtxt(shell_redshifts, delimiter=",")
        zmin = redshifts[shell_nr,0]
        zmax = redshifts[shell_nr,1]
        print(f"Reproducing shell {shell_nr} with zmin={zmin} and zmax={zmax}")
    else:
        zmin = None
        zmax = None
    zmin, zmax = comm.bcast((zmin, zmax))

    # Specify which particle types and properties to read in
    ptype = "Gas"
    property_names = ("Masses",)

    # Function to return the quantity to map, given the lightcone particle data dict
    def particle_mass(particle_data):
        return particle_data["Masses"]

    # Make the map
    map_data = smoothed_map.make_sky_map(input_filename, ptype, property_names, particle_mass,
                                         zmin, zmax, nside, smooth=True, progress=True)

    # Write out the new map
    dataset_name = "SmoothedGasMass"
    with h5py.File(output_filename, "w", driver="mpio", comm=comm) as outfile:
        phdf5.collective_write(outfile, dataset_name, map_data, comm)



if __name__ == "__main__":

    # Shell to do should be specified on the command line
    if comm_rank == 0:
        shell_nr = int(sys.argv[1])
    else:
        shell_nr = None
    shell_nr = comm.bcast(shell_nr)

    L1000N1800_HYDRO_FIDUCIAL_smoothed_gas_mass(shell_nr)
