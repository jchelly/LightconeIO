#!/bin/env python

import h5py
import lightcone_io.smoothed_map as smoothed_map
import virgo.mpi.parallel_hdf5 as phdf5

from mpi4py import MPI
comm = MPI.COMM_WORLD


def test_bh_map():

    nside = 16384
    input_filename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/sorted_lightcones_new/lightcone0_particles/lightcone0_0000.0.hdf5"
    output_filename = "/cosma8/data/dp004/jch/lightcone_map.hdf5"
    zmin, zmax = (0.05, 0.1)
    ptype = "BH"
    property_names = ("DynamicalMasses",)

    # Function to return the quantity to map, given the lightcone particle data dict
    def particle_mass(particle_data):
        return particle_data["DynamicalMasses"]

    dataset_name = "BlackHoleMass"

    # Make the map
    map_data = smoothed_map.make_full_sky_map(input_filename, ptype, property_names, particle_mass,
                                              zmin, zmax, nside, smooth=False)

    # Write out the new map
    with h5py.File(output_filename, "w", driver="mpio", comm=comm) as outfile:
        phdf5.collective_write(outfile, dataset_name, map_data, comm)


def test_gas_map():

    nside = 16384
    input_filename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/sorted_lightcones_new/lightcone0_particles/lightcone0_0000.0.hdf5"
    output_filename = "/cosma8/data/dp004/jch/lightcone_map.hdf5"
    zmin, zmax = (0.05, 0.1)
    ptype = "Gas"
    property_names = ("Masses",)

    # Function to return the quantity to map, given the lightcone particle data dict
    def particle_mass(particle_data):
        return particle_data["Masses"]

    dataset_name = "SmoothedGasMass"

    # Make the map
    map_data = smoothed_map.make_full_sky_map(input_filename, ptype, property_names, particle_mass,
                                              zmin, zmax, nside, smooth=True)

    # Write out the new map
    with h5py.File(output_filename, "w", driver="mpio", comm=comm) as outfile:
        phdf5.collective_write(outfile, dataset_name, map_data, comm)


def L1000N1800_HYDRO_FIDUCIAL_smoothed_gas_mass():

    nside = 16384
    input_filename = "/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"
    output_filename = "/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/new_maps/smoothed_gas_map.hdf5"
    zmin, zmax = (0.05, 0.1)
    ptype = "Gas"
    property_names = ("Masses",)

    # Function to return the quantity to map, given the lightcone particle data dict
    def particle_mass(particle_data):
        return particle_data["Masses"]

    dataset_name = "SmoothedGasMass"

    # Make the map
    map_data = smoothed_map.make_full_sky_map(input_filename, ptype, property_names, particle_mass,
                                              zmin, zmax, nside, smooth=True, progress=True)

    # Write out the new map
    with h5py.File(output_filename, "w", driver="mpio", comm=comm) as outfile:
        phdf5.collective_write(outfile, dataset_name, map_data, comm)



if __name__ == "__main__":

    L1000N1800_HYDRO_FIDUCIAL_smoothed_gas_mass()
    #test_bh_map()
