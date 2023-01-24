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

# May want to write one stderr file per MPI rank for debugging
#import sys
#sys.stderr = open(f"./stderr.{comm_rank}.log", "w")

projected_kernel = kernel.ProjectedKernel()

def native_endian(arr):
    return arr.astype(arr.dtype.newbyteorder("="))


def find_angular_smoothing_length(pos, hsml):
    """
    Return smoothing length in radians of the supplied particles,
    using small angle approximation for consistency with SWIFT
    """
    dist = np.sqrt(np.sum(pos**2, axis=1, dtype=float))
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

    # Normalize position vector
    part_pos = part_pos / np.sqrt(np.sum(part_pos**2, dtype=float))

    # Find radius containing the pixels to update
    angular_search_radius = angular_smoothing_length*kernel.kernel_gamma
            
    # Get pixel indexes to update
    pix_index = hp.query_disc(nside, part_pos, angular_search_radius)
    assert len(pix_index) >= 1

    # For each pixel, find angle between pixel centre and the particle
    pix_vec_x, pix_vec_y, pix_vec_z = hp.pixelfunc.pix2vec(nside, pix_index)
    dp = part_pos[0]*pix_vec_x + part_pos[1]*pix_vec_y + part_pos[2]*pix_vec_z
    dp = np.clip(dp, a_min=None, a_max=1.0)
    pix_angle = np.arccos(dp)

    # Evaluate the projected kernel for each pixel
    pix_weight = projected_kernel(pix_angle/angular_smoothing_length)

    # Normalize weights so that sum is one
    pix_val = part_val * pix_weight / np.sum(pix_weight, dtype=float)

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


def distribute_pixels(comm, nside):
    """
    Decide how to assign HEALPix pixels to MPI ranks.

    Here we assume the ring ordering scheme so that the minimum
    colatitude of a rank is the colatitude of its first pixel
    and the maximum colatitude of the rank is the colatitude of
    its last pixel, since all pixels in a ring have the same
    colatitude.

    We put equal numbers of pixels on each rank except that any
    leftovers are assigned to the last rank.

    Returns -
    
    nr_total_pixels: total number of pixels in the map
    nr_local_pixels: number of pixels on each rank
    local_offset:    offset between local and global pixel indexes
    min_theta:       minimum co-latitude theta on each rank
    max_theta:       minimum co-latitude theta on each rank
    """

    # Find number of pixels on each rank
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    nr_total_pixels = hp.pixelfunc.nside2npix(nside)
    nr_local_pixels = nr_total_pixels // comm_size
    local_offset = nr_local_pixels * comm_rank

    # Find range of pixels on each rank
    first_pixel = np.arange(comm_size, dtype=int) * nr_local_pixels
    last_pixel = first_pixel + nr_local_pixels - 1
    # Any extra pixels go on the last rank
    last_pixel[-1] = nr_total_pixels - 1
    nr_local_pixels = last_pixel[comm_rank] - first_pixel[comm_rank] + 1

    # Find range in colatitude on each rank:
    # Here we make an array with the value of theta at the boundary between
    # ranks, with com_size+1 elements. The first and last elements are 0 and 2pi.
    theta_boundary = np.ndarray(comm_size+1, dtype=float)
    theta_boundary[0] = 0.0
    theta_boundary[-1] = np.pi
    for i in range(0,comm_size-1):
        theta1, phi = hp.pixelfunc.pix2ang(nside, last_pixel[i])
        theta2, phi = hp.pixelfunc.pix2ang(nside, first_pixel[i+1])
        theta_boundary[i+1] = 0.5*(theta1+theta2)

    return nr_total_pixels, nr_local_pixels, local_offset, theta_boundary


def make_sky_map(input_filename, ptype, property_names, particle_value_function,
                      zmin, zmax, nside, vector = None, radius = None, smooth=True, progress=False):
    """
    Make a new HEALPix map from lightcone particle data

    input_filename: name of one file from the particle lightcone output
    ptype         : which particle type to use
    property_names: list of particle properties to read in
    particle_value_function: function which returns the quantity to map,
                    given a dict containing the arrays specified by property_names
    zmin          : minimum redshift to use
    zmax          : maximum redshift to use
    nside         : HEALPix resolution parameter
    smooth        : whether to smooth the map
    """
    
    if progress and comm_rank == 0:
        from tqdm import tqdm
        progress_bar = tqdm
    else:
        progress_bar = lambda x: x

    # Ensure property_names list contains coordinates and smoothing lengths
    property_names = list(property_names)
    if "Coordinates" not in property_names:
        property_names.append("Coordinates")
    if smooth and ("SmoothingLengths" not in property_names):
        property_names.append("SmoothingLengths")

    # Open the lightcone
    lightcone = pr.IndexedLightcone(input_filename, comm=comm)

    # Create an empty HEALPix map, distributed over MPI ranks.
    # Here we assume we can put equal sized chunks of the map on each rank.
    nr_total_pixels, nr_local_pixels, local_offset, theta_boundary = distribute_pixels(comm, nside)
    max_pixrad = hp.pixelfunc.max_pixrad(nside)
    message(f"Total number of pixels = {nr_total_pixels}")
    
    # Will read the full sky within the redshift range
    redshift_range = (zmin, zmax)

    # Determine number of particles to read on this MPI rank
    nr_particles_local = lightcone[ptype].count_particles(redshift_range=redshift_range,
                                                          vector=vector, radius=radius,)
    nr_particles_total = comm.allreduce(nr_particles_local)
    message(f"Total number of particles in selected cells = {nr_particles_total}")

    # Read in the particle data
    particle_data = lightcone[ptype].read_exact(property_names, vector, radius, redshift_range)
    nr_parts_tot = comm.allreduce(particle_data["Coordinates"].shape[0])
    message(f"Read in {nr_parts_tot} particles")

    # Find the particle positions and smoothing lengths
    part_pos_send = particle_data["Coordinates"]
    if smooth:
        part_hsml_send = find_angular_smoothing_length(part_pos_send, particle_data["SmoothingLengths"])
    else:
        part_hsml_send = np.zeros(part_pos_send.shape[0], dtype=float)

    # Find the quantities to add to the map
    part_val_send = unyt.unyt_array(particle_value_function(particle_data), dtype=float)
    val_total_global = comm.allreduce(np.sum(part_val_send, dtype=float))

    # Find units of the quantity which we're mapping
    map_units = part_val_send.units
    all_map_units = comm.allgather(map_units)
    for unit in all_map_units:
        if unit != map_units:
            raise RuntimeError("Quantity to map needs to have the same units on all MPI ranks!")

    # No longer need the particle data
    del particle_data

    # Determine range of colatitudes each particle will update.
    # Smoothing kernel drops to zero at kernel_gamma * smoothing length.
    # Note that particles with radius < max_pixrad can still update
    # pixels up to max_pixrad away because they update whatever pixel
    # they are in.
    radius = np.maximum(kernel.kernel_gamma*part_hsml_send, max_pixrad) # Might update pixels with centres within this radius
    theta, phi = hp.pixelfunc.vec2ang(part_pos_send)
    part_min_theta = np.clip(theta-radius, 0.0, np.pi) # Minimum central theta of pixels each particle might update
    part_max_theta = np.clip(theta+radius, 0.0, np.pi) # Maximum central theta of pixels each particle might update

    # Determine what range of MPI ranks each particle needs to be sent to
    part_first_rank = np.searchsorted(theta_boundary, part_min_theta, side="left") - 1
    part_first_rank = np.clip(part_first_rank, 0, comm_size-1)
    part_last_rank  = np.searchsorted(theta_boundary, part_max_theta, side="right") - 1
    part_last_rank  = np.clip(part_last_rank, 0, comm_size-1)
    assert np.all(theta_boundary[part_first_rank]  <= part_min_theta)
    assert np.all(theta_boundary[part_last_rank+1] >= part_max_theta)

    # Determine how many ranks each particle needs to be sent to
    nr_copies = part_last_rank - part_first_rank + 1
    assert np.all(nr_copies>=1) and np.all(nr_copies<=comm_size)

    # Duplicate the particles
    nr_parts = part_pos_send.shape[0]
    index = np.repeat(np.arange(nr_parts, dtype=int), nr_copies)
    part_pos_send  = part_pos_send[index,...]
    part_val_send  = part_val_send[index]
    part_hsml_send = part_hsml_send[index]
    del index

    # Determine destination rank for each particle copy
    nr_parts = np.sum(nr_copies)                # Total number of copied particles
    offset = np.cumsum(nr_copies) - nr_copies   # Offset to first copy of each particle in array of copies
    part_dest = -np.ones(nr_parts, dtype=int) # Destination rank for each copied particle
    for pfr, nrc, off in zip(part_first_rank, nr_copies, offset):
        part_dest[off:off+nrc] = np.arange(pfr, pfr+nrc, dtype=int)
    assert np.all(part_dest >=0) & np.all(part_dest<comm_size)
    message("Computed destination rank(s) for each particle")

    # Tidy up
    del offset
    del part_first_rank
    del part_last_rank
    del nr_copies

    # Copy particles to their destinations
    part_pos_recv, part_val_recv, part_hsml_recv = (
        exchange_particles(part_dest, (part_pos_send, part_val_send, part_hsml_send)))

    # Free send buffers
    del part_pos_send
    del part_val_send
    del part_hsml_send
    del part_dest

    # Allocate the output map
    map_data = unyt.unyt_array(np.zeros(nr_local_pixels, dtype=float), units=map_units)

    # Will use unit-less views to carry out the map update to minimize overhead
    map_view = map_data.ndarray_view()
    part_pos_recv_view = part_pos_recv.ndarray_view()
    part_val_recv_view = part_val_recv.ndarray_view()

    # Now each MPI rank has copies of all particles which affect its local
    # pixels. Process any particles which update single pixels.

    #single_pixel = part_hsml_recv*kernel.kernel_gamma < max_pixrad # Correct method
    single_pixel = part_hsml_recv < max_pixrad # To reproduce SWIFT bug (https://gitlab.cosma.dur.ac.uk/swift/swiftsim/-/merge_requests/1559)

    nr_single_pixel = comm.allreduce(np.sum(single_pixel))
    local_pix_index = hp.pixelfunc.vec2pix(nside, 
                                           part_pos_recv_view[single_pixel, 0],
                                           part_pos_recv_view[single_pixel, 1],
                                           part_pos_recv_view[single_pixel, 2]) - local_offset
    local = (local_pix_index >=0) & (local_pix_index < nr_local_pixels)
    np.add.at(map_view, local_pix_index[local], part_val_recv_view[single_pixel][local])
    del part_pos_recv_view
    del part_val_recv_view
    message(f"Applied {nr_single_pixel} single pixel updates")
    
    # Discard single pixel particles
    part_hsml_recv = part_hsml_recv[single_pixel==False]
    part_val_recv  = part_val_recv[single_pixel==False]
    part_pos_recv  = part_pos_recv[single_pixel==False,:]

    # Apply updates from particles which cover multiple pixels.
    #
    # Here we use ndarray views of the unyt part_val array and the map to avoid
    # any unit handling overhead. No conversion is necessary because arrays are 
    # in the same units.
    #
    # We only use the direction of the position vectors, so their units can
    # be ignored too.
    #
    nr_parts = len(part_val_recv)
    nr_parts_tot = comm.allreduce(nr_parts)
    part_pos_view  = part_pos_recv.ndarray_view()
    part_val_view  = part_val_recv.ndarray_view()
    for part_nr in progress_bar(range(nr_parts)):
        pix_index, pix_val = explode_particle(nside, part_pos_view[part_nr,:], part_val_view[part_nr], part_hsml_recv[part_nr])
        local_pix_index = pix_index - local_offset
        local = (local_pix_index >=0) & (local_pix_index < nr_local_pixels)
        # Don't need to use np.add.at here because pixel indexes are unique
        map_view[local_pix_index[local]] += pix_val[local]
    message(f"Applied {nr_parts_tot} multi-pixel updates")

    # Sanity check:
    # Sum over the map should equal sum of values to be accumulated to the map.
    map_sum = comm.allreduce(np.sum(map_data))
    ratio = map_sum / val_total_global
    message(f"Ratio (map sum / total values to add to map) = {ratio} (should be 1.0)")

    return map_data
