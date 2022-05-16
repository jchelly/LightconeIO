#!/bin/env python
#
# This script uses the lightcone particle data to create a new HEALPix map.
# Parallelised with MPI.

import healpy as hp
import numpy as np
import h5py

import lightcone_io.particle_reader as pr
import lightcone_io.kernel as kernel

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

def native_endian(arr):
    return arr.astype(arr.dtype.newbyteorder("="))

def exchange_particles(part_dest, arrays):

    # Sort arrays by destination
    send_arrays = []
    order = np.argsort(part_dest)
    for array in arrays:
        send_arrays.append(native_endian(array[order]))
    
    # Count particles to send to and receive from each rank
    send_count = np.bincount(part_dest, minlength=comm_size)
    recv_count = np.zeros_like(send_count)
    comm.Alltoall(send_count, recv_count)
    send_offset = np.cumsum(send_count) - send_count
    recv_offset = np.cumsum(recv_count) - recv_count
    nr_recv_tot = np.sum(recv_count)

    # Allocate receive buffers
    recv_arrays = []
    for array in send_arrays:
        recv_arrays.append(np.empty_like(array, shape=nr_recv_tot))

    # Exchange particles
    for sendbuf, recvbuf in zip(send_arrays, recv_arrays):
        psort.my_alltoallv(sendbuf, send_count, send_offset,
                           recvbuf, recv_count, recv_offset,
                           comm=comm)
    return recv_arrays


def message(m):
    comm.barrier()
    if comm_rank == 0:
        print(m)


def make_full_sky_map(input_filename, output_filename, zmin, zmax):
    
    # Open the lightcone
    lightcone = pr.IndexedLightcone(input_filename, comm=comm)
    print(lightcone["BH"].comm_rank, lightcone["BH"].comm_size)

    # Create an empty HEALPix map, distributed over MPI ranks.
    # Here we assume we can put equal sized chunks of the map on each rank.
    nside = 64
    npix = hp.pixelfunc.nside2npix(nside)
    if npix % comm_size != 0:
        raise RuntimeError("Map size must be a multiple of number of MPI ranks!")
    npix_local = npix // comm_size
    map_data = np.zeros(npix_local, dtype=float)
    message(f"Total number of pixels = {npix}")

    # Specify quantities to read in
    property_names = ("Coordinates", "DynamicalMasses", "ExpansionFactors")
    
    # Redshift range to read in
    redshift_range = (zmin, zmax)

    # Will read the full sky
    vector = None
    radius = None

    # Particle type
    ptype = "BH"

    # Determine number of particles to read on this MPI rank
    nr_particles_local = lightcone[ptype].count_particles(redshift_range=redshift_range,
                                                          vector=vector, radius=radius,)
    nr_particles_total = comm.allreduce(nr_particles_local)
    message(f"Total number of particles = {nr_particles_total}")
    print(f"Rank {comm_rank} has {nr_particles_local} particles to read")

    # Read in the particle data
    particle_data = lightcone[ptype].read_exact(property_names, vector, radius, redshift_range)
    message("Read in particle data")

    # Determine what pixel each particle is in
    pos = particle_data["Coordinates"].value # vec2pix can't handle unyt arrays, and we only need the direction from the vector anyway
    part_pix_send = hp.pixelfunc.vec2pix(nside, pos[:,0], pos[:,1], pos[:,2])

    # Decide which rank we need to send each particle to and its contribution to the map
    part_dest = part_pix_send // npix_local
    part_val_send  = particle_data["DynamicalMasses"]
    message("Computed destination rank for each particle")

    val_tot = comm.allreduce(np.sum(part_val_send, dtype=float))
    message(f"Total to add to map = {val_tot}")

    # Send each particle to the rank with the pixel it will update
    part_pix_recv, part_val_recv = exchange_particles(part_dest, (part_pix_send, part_val_send))
    message("Exchanged particles")

    # Add the received particles to the map
    pixel_nr = part_pix_recv - comm_rank * npix_local
    assert np.all(pixel_nr >=0) and np.all(pixel_nr < npix_local)
    np.add.at(map_data, pixel_nr, part_val_recv.value)
    message("Computed new map")
    
    # Write out the new map
    with h5py.File(output_filename, "w", driver="mpio", comm=comm) as outfile:
        phdf5.collective_write(outfile, "MassMap", map_data, comm)

    map_sum = comm.allreduce(np.sum(map_data))
    message(f"Wrote map, sum = {map_sum}.")


if __name__ == "__main__":

    # Specify one file from the spatially indexed lightcone particle data
    input_filename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/sorted_lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"

    # Where to write the new HEALPix map
    output_filename = "/cosma8/data/dp004/jch/lightcone_map.hdf5"

    # Redshift range to do
    zmin, zmax = (0.0, 0.05)

    make_full_sky_map(input_filename, output_filename, zmin, zmax)
