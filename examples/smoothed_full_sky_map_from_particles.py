#!/bin/env python
#
# This script uses the lightcone particle data to create a new HEALPix map.
# Parallelised with MPI.

import healpy as hp
import numpy as np
import h5py
import unyt

import lightcone_io.particle_reader as pr
import lightcone_io.kernel as kernel

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

projected_kernel = kernel.ProjectedKernel()

def native_endian(arr):
    return arr.astype(arr.dtype.newbyteorder("="))


def find_angular_smoothing_length(pos, hsml):
    """
    Return smoothing length in radians of the supplied particles,
    using small angle approximation for consistency with SWIFT
    """
    dist = np.sqrt(np.sum(pos**2, axis=1))
    small_angle = dist > 10.0*hsml
    large_angle = (small_angle==False)
    angular_smoothing_length = np.ndarray(pos.shape[0], dtype=float)
    ratio = (hsml/dist).to(unyt.dimensionless)
    angular_smoothing_length[small_angle] = ratio[small_angle]
    angular_smoothing_length[large_angle] = np.arctan(ratio[large_angle])
    return angular_smoothing_length


def explode_particle(nside, part_pos, part_val, angular_smoothing_length):
    """
    Given a particle's position vector and angular radius,
    return indexes of the pixels it will update and the values
    to add to the pixels.

    part_pos: particle's position vector
    part_val: particle's contribution to the map
    angular_smoothing_length: smoothing length in radians

    Returns

    pix_index: array of indexes of the pixels to update
    pix_val: array of values to add to the pixels
    """

    # Normalize position vector and strip units
    part_pos = part_pos.value / np.sqrt(np.sum(part_pos.value**2))

    # Find radius containing the pixels to update
    angular_search_radius = angular_smoothing_length*kernel.kernel_gamma
            
    # Get pixel indexes to update
    pix_index = hp.query_disc(nside, part_pos, angular_search_radius)

    # For each pixel, find angle between pixel centre and the particle
    pix_vec  = np.column_stack(hp.pixelfunc.pix2vec(nside, pix_index))
    dp = np.sum(part_pos[None,:]*pix_vec, axis=1)
    dp[dp > 1.0] = 1.0 # In case of rounding error
    pix_angle = np.arccos(dp)

    # Evaluate the projected kernel for each pixel
    pix_weight = projected_kernel(pix_angle/angular_smoothing_length)

    # Normalize weights so that sum is one
    pix_weight = pix_weight / np.sum(pix_weight)
    pix_val = (part_val * pix_weight).to(part_val.units)

    return pix_index, pix_val


def exchange_particles(part_dest, arrays):

    # Sort arrays by destination
    send_arrays = []
    order = np.argsort(part_dest)
    for array in arrays:
        send_arrays.append(native_endian(array[order,...]))
    
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
        shape = list(array.shape)
        shape[0] = nr_recv_tot
        recv_arrays.append(np.empty_like(array, shape=shape))

    # Exchange particles
    for sendbuf, recvbuf in zip(send_arrays, recv_arrays):
        if len(sendbuf.shape) == 1:
            # 1D array
            psort.my_alltoallv(sendbuf, send_count, send_offset,
                               recvbuf, recv_count, recv_offset,
                               comm=comm)
        elif len(sendbuf.shape) == 2:
            # 2D array: flatten before sending
            ndims = sendbuf.shape[1]
            psort.my_alltoallv(sendbuf.ravel(), ndims*send_count, ndims*send_offset,
                               recvbuf.ravel(), ndims*recv_count, ndims*recv_offset,
                               comm=comm)
        else:
            raise RuntimeError("Can only exchange arrays with 1 or 2 dimensions")

    return recv_arrays


def create_empty_array(arr, comm):
    """
    Ensure that if arr is None on some ranks it is replaced with a zero
    sized array of the same dtype and units as the other ranks.
    """
    has_units = comm.allreduce(hasattr(arr, "units"), op=MPI.MAX)
    local_dtype = None if arr is None else arr.dtype
    if has_units:
        local_units = None if arr is None else arr.units
    else:
        local_units = None
    local_shape = None if arr is None else arr.shape[1:]
    dtypes = comm.allgather(local_dtype)
    units  = comm.allgather(local_units)
    shapes = comm.allgather(local_shape)
    if arr is None:
        for dt, u, s in zip(dtypes, units, shapes):
            if dt is not None:
                arr = np.ndarray((0,)+s, dtype=dt)
                if has_units:
                    arr = unyt.unyt_array(arr, units=u, dtype=dt)
                return arr
        raise ValueError("All input arrays are None in create_empty_array()!")
    else:
        return arr


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
    max_pixrad = hp.pixelfunc.max_pixrad(nside)
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

    # Fake BH smoothing lengths for testing
    nr_parts = particle_data["Coordinates"].shape[0]
    fake_hsml = np.ones(nr_parts, dtype=float) * 0.1
    fake_hsml = unyt.unyt_array(fake_hsml, units=particle_data["Coordinates"].units)
    particle_data["SmoothingLengths"] = fake_hsml

    # Find the particle positions and smoothing lengths
    part_pos_send = particle_data["Coordinates"]
    part_hsml_send = find_angular_smoothing_length(part_pos_send, particle_data["SmoothingLengths"])
    # Determine what pixel each particle is in
    part_pix_send = hp.pixelfunc.vec2pix(nside, part_pos_send[:,0].value, part_pos_send[:,1].value, part_pos_send[:,2].value)

    # Decide which rank we need to send each particle to and its contribution to the map
    part_dest = part_pix_send // npix_local
    part_val_send  = particle_data["DynamicalMasses"]
    message("Computed destination rank for each particle")

    # Report total to be added to the map
    val_total_local = np.sum(part_val_send, dtype=float)
    val_total_global = comm.allreduce(val_total_local)
    message(f"Total to add to the map = {val_total_global}")

    # Send each particle to the rank with the central pixel it will update
    part_pix_recv, part_val_recv, part_hsml_recv, part_pos_recv = (
        exchange_particles(part_dest, (part_pix_send, part_val_send, part_hsml_send, part_pos_send)))
    message("Exchanged particles")

    #
    # Tidy up to save memory
    #
    del particle_data
    del part_dest
    del part_pix_send
    del part_val_send
    del part_hsml_send
    del part_pos_send

    #
    # Now we have each particle stored on the rank which has the pixel
    # its centre is in. But some particles will update multiple pixels.
    # First, process any particles which update single pixels.
    # 
    single_pixel = part_hsml_recv < max_pixrad
    local_pix_index = part_pix_recv[single_pixel] - comm_rank * npix_local
    assert np.all(local_pix_index >=0) and np.all(local_pix_index < npix_local)
    np.add.at(map_data, local_pix_index, part_val_recv[single_pixel].value)
    message("Applied single pixel updates")
    
    # Discard single pixel particles
    part_hsml_recv = part_hsml_recv[single_pixel==False]
    part_pix_recv  = part_pix_recv[single_pixel==False]
    part_val_recv  = part_val_recv[single_pixel==False]
    part_pos_recv  = part_pos_recv[single_pixel==False,:]

    #
    # Apply local updates from particles which cover multiple pixels.
    # Also accumulates list of non-local updates to send to other ranks.
    #
    non_local = []
    nr_part = len(part_pix_recv)
    nr_local_updates = comm.allreduce(nr_part)
    for part_nr in range(nr_part):
        pix_index, pix_val = explode_particle(nside, part_pos_recv[part_nr,:], part_val_recv[part_nr], part_hsml_recv[part_nr])
        local_pix_index = pix_index - comm_rank * npix_local
        local = (local_pix_index >= 0) & (local_pix_index < npix_local)
        np.add.at(map_data, local_pix_index[local], pix_val[local].value)
        # Store non-local updates to apply later
        if np.sum(local==False) > 0:
            non_local.append((pix_index[local==False], pix_val[local==False]))
    message(f"Applied local multi-pixel updates for {nr_local_updates} particles")

    #
    # Exchange non-local updates, if there are any. These are cases where a
    # particle on this rank needs to update pixels stored on another rank.
    #
    nr_non_local_tot = comm.allreduce(len(non_local))
    if nr_non_local_tot > 0:
        # Make arrays of updates to do
        if len(non_local) > 0:
            pix_index_send = np.concatenate([nl[0] for nl in non_local])
            pix_val_send = np.concatenate([nl[1] for nl in non_local])
        else:
            pix_index_send = None
            pix_val_send = None
        # On ranks with no updates, create zero sized arrays of the appropriate type
        pix_index_send = create_empty_array(pix_index_send, comm)
        pix_val_send   = create_empty_array(pix_val_send, comm)
        nr_nonlocal_updates = len(pix_val_send)
        # Send updates to the rank with the pixel they should be applied to
        pix_dest = pix_index_send // npix_local
        pix_index_recv, pix_val_recv = exchange_particles(pix_dest, (pix_index_send, pix_val_send))
        # Apply the imported updates to the local part of the map
        local_pix_index = pix_index_recv - comm_rank * npix_local
        assert np.all(local_pix_index>=0) and np.all(local_pix_index < npix_local)
        np.add.at(map_data, local_pix_index, pix_val_recv.value)
    else:
        # No ranks have any non-local updates to do
        nr_nonlocal_updates = 0
    nr_nonlocal_updates = comm.allreduce(nr_nonlocal_updates)
    message(f"Applied {nr_nonlocal_updates} non-local multi-pixel updates")

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
