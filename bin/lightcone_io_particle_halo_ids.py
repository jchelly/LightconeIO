#!/bin/env python

import os
import sys
import argparse
import time
t0 = time.time()

import numpy as np
import h5py
import scipy.spatial

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

import virgo.util.match as match
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
import virgo.mpi.util as mpi_util


# Constants to identify methods for dealing with particles in multiple halos
FRACTIONAL_RADIUS=0
MOST_MASSIVE=1
LEAST_MASSIVE=2
overlap_methods = {
    "fractional_radius" : FRACTIONAL_RADIUS,
    "most_massive"      : MOST_MASSIVE,
    "least_massive"     : LEAST_MASSIVE,
    }


def message(m):
    if comm_rank == 0:
        t1 = time.time()
        elapsed = t1-t0
        print(f"{elapsed:.1f}s: {m}")


def read_lightcone_halo_positions_and_radii(args, radius_name, mass_name):
    """
    Read in the lightcone halo catalogue and cross reference with SOAP
    to find the SO radius and mass for each halo in the lightcone.

    Assumes that positions in the lightcone are comoving and in the
    same units as SOAP (except for the expansion factor dependence).
    """

    # Parallel read the halo catalogue: need (x,y,z), snapnum, id
    message("Reading lightcone halo catalogue")
    halo_lightcone_datasets = ("LightconeXcminpot", "LightconeYcminpot", "LightconeZcminpot", "SnapNum", "ID")
    mf = phdf5.MultiFile(args.halo_lightcone_filenames, file_nr_attr=("Header", "NumberOfFiles"), comm=comm)
    halo_lightcone_data = mf.read(halo_lightcone_datasets, group="Subhalo", read_attributes=True)

    # Store index in halo lightcone of each halo
    nr_local_halos = len(halo_lightcone_data["ID"])
    offset = comm.scan(nr_local_halos) - nr_local_halos
    halo_lightcone_data["IndexInHaloLightcone"] = np.arange(nr_local_halos, dtype=int) + offset

    # Repartition halos for better load balancing
    message("Repartition halo catalogue")
    nr_local_halos = len(halo_lightcone_data["ID"])
    nr_total_halos = comm.allreduce(nr_local_halos)
    nr_desired = np.zeros(comm_size, dtype=int)
    nr_desired[:] = nr_total_halos // comm_size
    nr_desired[:nr_total_halos % comm_size] += 1
    assert np.sum(nr_desired) == nr_total_halos
    for name in halo_lightcone_data:
        halo_lightcone_data[name] = psort.repartition(halo_lightcone_data[name], nr_desired, comm=comm)

    # Merge x/y/z into a single array
    halo_pos = np.column_stack((halo_lightcone_data["LightconeXcminpot"],
                                halo_lightcone_data["LightconeYcminpot"],
                                halo_lightcone_data["LightconeZcminpot"]))
    del halo_lightcone_data["LightconeXcminpot"]
    del halo_lightcone_data["LightconeYcminpot"]
    del halo_lightcone_data["LightconeZcminpot"]
    halo_lightcone_data["Pos_minpot"] = halo_pos 
    del halo_pos

    # The input catalogue is ordered by redshift, but we want a mix of redshifts on each rank
    message("Reassign halos to MPI ranks")
    nr_local_halos = len(halo_lightcone_data["ID"])
    rng = np.random.default_rng()
    sort_key = rng.integers(comm_size, size=nr_local_halos, dtype=np.int32)
    order = psort.parallel_sort(sort_key, comm=comm, return_index=True)
    for name in sorted(halo_lightcone_data):
        psort.fetch_elements(halo_lightcone_data[name], order, result=halo_lightcone_data[name], comm=comm)

    # Sort locally by snapnum
    message("Sorting local lightcone halos by snapshot")
    order = np.argsort(halo_lightcone_data["SnapNum"])
    for name in halo_lightcone_data:
        halo_lightcone_data[name] = halo_lightcone_data[name][order,...]

    # Find range of local halos at each snapshot
    message("Identifying halos at each snapshot")
    unique_snap, snap_offset, snap_count = np.unique(halo_lightcone_data["SnapNum"],
                                                     return_index=True, return_counts=True)

    # Find full range of snapshots across all MPI ranks
    min_snap = comm.allreduce(np.amin(unique_snap), op=MPI.MIN)
    max_snap = comm.allreduce(np.amax(unique_snap), op=MPI.MAX)

    # Make snapnum, count and offset arrays which include snapshots not present on this rank:
    # We're going to do collective reads of the SOAP outputs so all ranks need to agree on
    # what range of snapshots to do.
    nr_snaps = max_snap - min_snap + 1
    unique_snap_all = np.arange(min_snap, max_snap+1, dtype=int)
    snap_offset_all = np.zeros(nr_snaps, dtype=int)
    snap_count_all  = np.zeros(nr_snaps, dtype=int)
    for us, so, sc in zip(unique_snap, snap_offset, snap_count):
        i = us - min_snap
        assert unique_snap_all[i] == us
        snap_offset_all[i] = so
        snap_count_all[i] = sc

    # Allocate storage for radius of each lightcone halo
    nr_halos = len(halo_lightcone_data["ID"])
    halo_lightcone_data[radius_name] = None # Don't know dtype for radius array yet

    # Loop over snapshots
    for snapnum in unique_snap_all:

        # Datasets to read from SOAP
        soap_datasets = ("VR/ID", radius_name, mass_name)

        # Read the SOAP catalogue for this snapshot
        message(f"Reading SOAP output for snapshot {snapnum}")
        mf = phdf5.MultiFile(args.soap_filenames % {"snap_nr" : snapnum}, file_idx=(0,), comm=comm)
        soap_data = mf.read(soap_datasets, read_attributes=True)

        # Get the expansion factor of this snapshot
        if comm_rank == 0:
            with h5py.File(args.soap_filenames % {"snap_nr" : snapnum}, "r") as infile:
                a = float(infile["SWIFT"]["Header"].attrs["Scale-factor"])
        else:
            a = None
        a = comm.bcast(a)

        # Ensure radii are in comoving units
        radius_a_exponent = float(soap_data[radius_name].attrs["a-scale exponent"])
        soap_data[radius_name] *= a**(radius_a_exponent-1.0)

        # Match lightcone halos at this snapshot to SOAP halos by ID
        message("Finding lightcone halos in SOAP output")
        i1 = snap_offset_all[snapnum-min_snap]
        i2 = snap_offset_all[snapnum-min_snap] + snap_count_all[snapnum-min_snap]
        assert np.all(halo_lightcone_data["SnapNum"][i1:i2] == snapnum)
        ptr = psort.parallel_match(halo_lightcone_data["ID"][i1:i2], soap_data["VR/ID"], comm=comm)
        assert np.all(ptr>=0) # All halos in the lightcone should be found in SOAP

        # Allocate storage for radii now that we know what dtype SOAP uses
        if halo_lightcone_data[radius_name] is None:
            radius_dtype = soap_data[radius_name].dtype
            halo_lightcone_data[radius_name] = phdf5.AttributeArray(-np.ones(nr_halos, dtype=radius_dtype),
                                                                    attrs=soap_data[radius_name].attrs)
            mass_dtype = soap_data[mass_name].dtype
            halo_lightcone_data[mass_name] = phdf5.AttributeArray(-np.ones(nr_halos, dtype=mass_dtype),
                                                                  attrs=soap_data[mass_name].attrs)

        message("Storing SO radii for lightcone halos at this snapshot")
        psort.fetch_elements(soap_data[radius_name], ptr,
                             result=halo_lightcone_data[radius_name][i1:i2], comm=comm)

        message("Storing SO masses for lightcone halos at this snapshot")
        psort.fetch_elements(soap_data[mass_name], ptr,
                             result=halo_lightcone_data[mass_name][i1:i2], comm=comm)

    # All halos should have been assigned a radius
    assert np.all(halo_lightcone_data[radius_name] >= 0)
    assert np.all(halo_lightcone_data[mass_name] >= 0)

    return halo_lightcone_data


def read_lightcone_index(args):
    """
    Read the index file and determine names of all particle files
    and which particle types are present
    """
    
    # Particle types which may be in the lightcone:
    type_names = ("BH", "DM", "Gas", "Neutrino", "Stars")
    type_z_range = {}

    # Now, find the lightcone particle output and read the index info
    index_file = args.lightcone_dir+"/"+args.lightcone_base+"_index.hdf5"
    if comm_rank == 0:
        with h5py.File(index_file, "r") as index:
            lc = index["Lightcone"]
            nr_mpi_ranks = int(lc.attrs["nr_mpi_ranks"])
            final_file_on_rank = lc.attrs["final_particle_file_on_rank"]
            for tn in type_names:
                min_z = float(lc.attrs["minimum_redshift_"+tn])
                max_z = float(lc.attrs["maximum_redshift_"+tn])
                if max_z > min_z:
                    type_z_range[tn] = (min_z, max_z)
    else:
        nr_mpi_ranks = None
        final_file_on_rank = None
        type_z_range = None
    nr_mpi_ranks, final_file_on_rank, type_z_range = comm.bcast((nr_mpi_ranks, final_file_on_rank, type_z_range))

    # Report which particle types we found
    for name in type_z_range:
        min_z, max_z = type_z_range[name]
        message(f"have particles for type {name} from z={min_z} to z={max_z}")

    # Make a full list of files to read
    all_particle_files = []
    for rank_nr in range(nr_mpi_ranks):
        for file_nr in range(final_file_on_rank[rank_nr]+1):
            filename = f"{args.lightcone_dir}/{args.lightcone_base}_particles/{args.lightcone_base}_{file_nr:04d}.{rank_nr}.hdf5"
            all_particle_files.append(filename)

    return type_z_range, all_particle_files


def compute_particle_group_index(halo_id, halo_pos, halo_radius, halo_mass, part_pos,
                                 overlap_method):
    """
    Tag particles which are within the SO radius of a halo
    """

    # Assign indexes to the particles so we can restore their ordering later
    nr_particles = part_pos.shape[0]
    offset = comm.scan(nr_particles) - nr_particles
    part_index = np.arange(nr_particles, dtype=np.int64) + offset

    nr_halos = halo_pos.shape[0]
    nr_halos_total = comm.allreduce(nr_halos)
    nr_particles_total = comm.allreduce(nr_particles)
    message(f"Have {nr_particles_total} particles and {nr_halos_total} halos")

    # Will split the halos and particles by x coordinate, with a roughly
    # constant number of particles per rank. First, sort the particles by x.
    message("Sorting particles by x coordinate")
    sort_key = part_pos[:,0].copy()
    order = psort.parallel_sort(sort_key, return_index=True, comm=comm)
    del sort_key
    psort.fetch_elements(part_pos, order, result=part_pos, comm=comm)
    psort.fetch_elements(part_index, order, result=part_index, comm=comm)
    del order

    # Find the maximum halo radius
    local_max_radius = np.amax(halo_radius)
    max_radius = comm.allreduce(local_max_radius, op=MPI.MAX)

    # Find maximum distance to any particle
    local_max_particle_distance = np.amax(np.sqrt(np.sum(part_pos**2, axis=1)))
    max_particle_distance = comm.allreduce(local_max_particle_distance, op=MPI.MAX)

    # Find the subset of halos which can overlap the particle distribution:
    # This helps in case the halo lightcone goes out to much higher redshift
    # than the particle lightcone.
    halo_distance = np.sqrt(np.sum(halo_pos**2, axis=1))
    within_distance = halo_distance < (max_particle_distance + max_radius)
    halo_pos = halo_pos[within_distance,:]
    halo_id = halo_id[within_distance]
    halo_radius = halo_radius[within_distance]
    halo_mass = halo_mass[within_distance]
    nr_halos_left = comm.allreduce(halo_id.shape[0], op=MPI.SUM)
    message(f"Halos within redshift range = {nr_halos_left} of {nr_halos_total}")

    # Determine the range of x coordinates of halos which could overlap particles on this rank
    local_x_min = np.amin(part_pos[:,0]) - max_radius
    x_min_on_rank = np.asarray(comm.allgather(local_x_min), dtype=part_pos.dtype)
    local_x_max = np.amax(part_pos[:,0]) + max_radius
    x_max_on_rank = np.asarray(comm.allgather(local_x_max), dtype=part_pos.dtype)

    # Sort local halos by x coordinate
    message("Sorting local lightcone halos by x coordinate")
    order = np.argsort(halo_pos[:,0])
    halo_pos = halo_pos[order,:]
    halo_id = halo_id[order]
    halo_radius = halo_radius[order]
    halo_mass = halo_mass[order]
    del order

    # Determine what range of halos needs to be sent to each MPI rank
    first_halo_for_rank = np.searchsorted(halo_pos[:,0], x_min_on_rank, side="left")
    last_halo_for_rank = np.searchsorted(halo_pos[:,0], x_max_on_rank, side="right")
    nr_halos_for_rank = last_halo_for_rank - first_halo_for_rank
    nr_halos_for_rank_total = comm.allreduce(nr_halos_for_rank)
    total_nr_halos_read = comm.allreduce(halo_pos.shape[0])
    total_nr_halos_sent = np.sum(nr_halos_for_rank_total)
    duplication_factor = total_nr_halos_sent / total_nr_halos_read
    message(f"Minimum halos on rank after exchange = {np.amin(nr_halos_for_rank_total)}")
    message(f"Maximum halos on rank after exchange = {np.amax(nr_halos_for_rank_total)}")
    message(f"Duplication factor = {duplication_factor}")

    # Compute lengths and offsets for alltoallv halo exchange
    send_offset = first_halo_for_rank
    send_count = nr_halos_for_rank
    recv_count = np.asarray(comm.alltoall(send_count), dtype=send_count.dtype)
    recv_offset = np.cumsum(recv_count) - recv_count

    # Exchange halo IDs
    message("Exchanging halo IDs")
    halo_id_recv = np.empty_like(halo_id, shape=np.sum(recv_count))
    psort.my_alltoallv(halo_id, send_count, send_offset,
                       halo_id_recv, recv_count, recv_offset,
                       comm=comm)
    halo_id = halo_id_recv
    del halo_id_recv
    comm.barrier()

    # Exchange halo radii
    message("Exchanging halo radii")
    halo_radius_recv = np.empty_like(halo_radius, shape=np.sum(recv_count))
    psort.my_alltoallv(halo_radius, send_count, send_offset,
                       halo_radius_recv, recv_count, recv_offset,
                       comm=comm)
    halo_radius = halo_radius_recv
    del halo_radius_recv
    comm.barrier()

    # Exchange halo masses
    message("Exchanging halo masses")
    halo_mass_recv = np.empty_like(halo_mass, shape=np.sum(recv_count))
    psort.my_alltoallv(halo_mass, send_count, send_offset,
                       halo_mass_recv, recv_count, recv_offset,
                       comm=comm)
    halo_mass = halo_mass_recv
    del halo_mass_recv
    comm.barrier()
    
    # Exchange halo positions:
    # These are vectors so flatten, exchange then restore shape
    message("Exchanging halo positions")
    halo_pos.shape = (-1,)
    halo_pos_recv = np.empty_like(halo_pos, shape=3*np.sum(recv_count))
    psort.my_alltoallv(halo_pos, send_count*3, send_offset*3,
                       halo_pos_recv, recv_count*3, recv_offset*3,
                       comm=comm)
    halo_pos = halo_pos_recv
    halo_pos.shape = (-1, 3)
    del halo_pos_recv
    comm.barrier()

    # Build a kdtree with the local particles
    message("Building kdtree")
    tree = scipy.spatial.KDTree(part_pos)

    # Allocate output array for the particle halo IDs etc
    nr_parts = part_pos.shape[0]
    part_halo_id = -np.ones(nr_parts, dtype=np.int64)         # ID of halo particle is assigned to
    part_halo_mass = -np.ones(nr_parts, dtype=np.float32)     # Mass of the halo
    part_halo_r_frac_2 = -np.ones(nr_parts, dtype=np.float32) # Smallest ((Particle radius)/(halo r200))**2 so far

    # Report maximum halo radius
    max_radius = comm.allreduce(np.amax(halo_radius), op=MPI.MAX)
    message(f"Maximum halo radius = {max_radius}")

    # Loop over local halos
    nr_assigned = 0
    message("Assigning halo IDs to particles")
    for i in range(len(halo_id)):
            
        # Identify particles within this halo's radius
        idx = np.asarray(tree.query_ball_point(halo_pos[i,:], halo_radius[i]), dtype=int)

        # Compute radius squared for each particle
        r_part_2 = np.sum((part_pos[idx,:] - halo_pos[i,:])**2.0, axis=1)
        
        # Compute ((particle radius)/(halo radius))**2
        r_frac_2 = r_part_2 / (halo_radius[i]**2)

        # Identify particles to update
        if overlap_method == FRACTIONAL_RADIUS:
            # Assign particles to this halo if (particle radius)/(halo radius) is smaller
            # than the smallest value so far
            to_update = (r_frac_2 < part_halo_r_frac_2[idx]) | (part_halo_r_frac_2[idx] < 0.0)
        elif overlap_method == MOST_MASSIVE:
            # Assign particles to this halo if this is the most massive halo the particle
            # has been found to be in so far
            to_update = halo_mass[i] > part_halo_mass[idx]
        elif overlap_method == LEAST_MASSIVE:
            # Assign particles to this halo if this is the least massive halo the particle
            # has been found to be in so far
            to_update = (halo_mass[i] < part_halo_mass[idx]) | (part_halo_mass[idx] < 0.0)
        else:
            raise ValueError("Unrecognized value of overlap_method")        
        idx = idx[to_update]

        # Tag particles to update with the halo ID, mass and fractional radius
        part_halo_id[idx]       = halo_id[i]
        part_halo_mass[idx]     = halo_mass[i]
        part_halo_r_frac_2[idx] = r_frac_2[to_update]
        nr_assigned            += len(idx)

    nr_assigned_tot = comm.allreduce(nr_assigned)
    fraction_assigned = nr_assigned_tot / nr_particles_total
    message(f"Total particles assigned to halos = {nr_assigned_tot}")
    message(f"Fraction assigned = {fraction_assigned} (inc. duplicates due to halo overlap)")

    # Tidy up
    del halo_id
    del halo_pos
    del halo_radius
    del halo_mass
    del part_pos

    # Return r_frac=r/r200 for particles in halos and -1 for those not in halos
    in_halo = (part_halo_id >= 0)
    part_halo_r_frac = np.where(in_halo, np.sqrt(part_halo_r_frac_2), -1.0)
    del part_halo_r_frac_2

    # Restore original particle ordering and return halo IDs etc
    message("Restoring particle order")
    order = psort.parallel_sort(part_index, return_index=True, comm=comm)
    del part_index
    psort.fetch_elements(part_halo_id, order, result=part_halo_id, comm=comm)
    psort.fetch_elements(part_halo_mass, order, result=part_halo_mass, comm=comm)
    psort.fetch_elements(part_halo_r_frac, order, result=part_halo_r_frac, comm=comm)

    return part_halo_id, part_halo_mass, part_halo_r_frac


def main(args):

    # Determine method to deal with overlapping halos
    overlap_method = overlap_methods[args.overlap_method]
    
    # Read in position and radius for halos in the lightcone
    radius_name = f"{args.soap_so_name}/SORadius"
    mass_name = f"{args.soap_so_name}/TotalMass"
    halo_lightcone_data = read_lightcone_halo_positions_and_radii(args, radius_name, mass_name)

    # Locate the particle data
    type_z_range, all_particle_files = read_lightcone_index(args)

    # Generate filenames for the output:
    # These are the input filenames with the directory replaced with argument output_dir.
    output_filenames = []
    for input_filename in all_particle_files:
        dirname, filename = os.path.split(input_filename)
        output_filenames.append(os.path.join(args.output_dir, filename))

    # Open the input particle file set
    mf = phdf5.MultiFile(all_particle_files, comm=comm)

    # Loop over types to do
    create_files = True
    for ptype in type_z_range:
        
        message(f"Processing particle type {ptype}")

        # Read in positions of lightcone particles of this type
        message(f"Reading particles")
        part_pos = mf.read("Coordinates", group=ptype)

        # Record number of particles read from each file
        elements_per_file = mf.get_elements_per_file("Coordinates", group=ptype)

        # Rebalance particle load between MPI ranks
        nr_parts_per_rank_read = np.asarray(comm.allgather(part_pos.shape[0]), dtype=int)
        nr_parts_total = np.sum(nr_parts_per_rank_read)
        nr_parts_per_rank_balanced = np.zeros_like(nr_parts_per_rank_read)
        nr_av = (nr_parts_total // comm_size)
        nr_parts_per_rank_balanced[:] = nr_av
        nr_parts_per_rank_balanced[:nr_parts_total % comm_size] += 1
        assert np.sum(nr_parts_per_rank_balanced) == nr_parts_total
        part_pos = psort.repartition(part_pos, ndesired=nr_parts_per_rank_balanced, comm=comm)

        # Report load balancing
        max_nr_parts = comm.allreduce(part_pos.shape[0], op=MPI.MAX)
        min_nr_parts = comm.allreduce(part_pos.shape[0], op=MPI.MIN)
        message(f"No. of particles per rank min={min_nr_parts}, max={max_nr_parts}")

        # Assign group indexes to the particles
        message("Assigning group indexes")
        halo_id = halo_lightcone_data["IndexInHaloLightcone"]
        halo_pos = halo_lightcone_data["Pos_minpot"]
        halo_radius = halo_lightcone_data[radius_name]
        halo_mass = halo_lightcone_data[mass_name]
        part_halo_id, part_halo_mass, part_halo_r_frac = compute_particle_group_index(halo_id, halo_pos, halo_radius, halo_mass, part_pos, overlap_method)
        del part_pos
        del halo_id
        del halo_pos
        del halo_radius
        del halo_mass

        # Restore original partitioning of particles
        part_halo_id = psort.repartition(part_halo_id, ndesired=nr_parts_per_rank_read, comm=comm)
        part_halo_mass = psort.repartition(part_halo_mass, ndesired=nr_parts_per_rank_read, comm=comm)
        part_halo_r_frac = psort.repartition(part_halo_r_frac, ndesired=nr_parts_per_rank_read, comm=comm)

        # Write the output, appending to file if this is not the first particle type
        message(f"Writing output to {args.output_dir}")
        mode = "w" if create_files else "r+"
        datasets = {
            "IndexInHaloLightcone" : part_halo_id,
            "FractionalRadius" : part_halo_r_frac,
            "HaloMass" : part_halo_mass,
        }
        attributes = {
            "IndexInHaloLightcone" : halo_lightcone_data["ID"].attrs,
            "FractionalRadius" : halo_lightcone_data["ID"].attrs,
            "HaloMass" : halo_lightcone_data[mass_name].attrs,
        }
        mf.write(datasets, elements_per_file, output_filenames, mode,
                 group=ptype, attrs=attributes, gzip=6, shuffle=True)
                
        # Tidy up before reading next particle type
        del part_halo_id
        del part_halo_r_frac
        del part_halo_mass

        # Only need to create new output files for the first type
        create_files = False
        
    comm.barrier()

    # Discard reordered halo lightcone data
    del halo_lightcone_data
    del mf

    #
    # Now, for each particle in the lightcone we have the index in the halo
    # lightcone of the halo it belongs to.
    #
    # Next we re-read the halo lightcone to copy any extra halo properties we
    # want to store for each particle. This uses less memory than passing all
    # quantities through the calculation above.
    #
    message("Reading lightcone halo properties to copy to output particle files")
    halo_properties = (
        "Subhalo/ID",
        "Subhalo/SnapNum",
    )
    mf_in = phdf5.MultiFile(args.halo_lightcone_filenames, file_nr_attr=("Header", "NumberOfFiles"), comm=comm)
    halo_lightcone_data = mf_in.read(halo_properties, read_attributes=True)
    halo_lightcone_data["Subhalo/ID"] = halo_lightcone_data["Subhalo/ID"].astype(np.int64) # Avoid using unsigned int

    # Open the set of particle files to update
    mf_out = phdf5.MultiFile(output_filenames, comm=comm)

    # Loop over particle types to update
    for ptype in type_z_range:
   
        message(f"Reading halo index for particles of type {ptype}")
        halo_index = mf_out.read(f"{ptype}/IndexInHaloLightcone")
        elements_per_file = mf_out.get_elements_per_file(f"{ptype}/IndexInHaloLightcone")
        
        for prop_name in halo_properties:
            message(f"Pass through quantity {prop_name} for type {ptype}")
            in_halo = (halo_index >= 0)
            dtype = halo_lightcone_data[prop_name].dtype
            shape = (halo_index.shape[0],)+halo_lightcone_data[prop_name].shape[1:]
            prop_data = -np.ones(shape, dtype=dtype) # Set property=-1 if particle not in halo
            prop_data[in_halo,...] = psort.fetch_elements(halo_lightcone_data[prop_name], halo_index[in_halo], comm=comm)
            # Write the new dataset to the output files
            dataset_name = f"{ptype}/{prop_name.split('/')[-1]}"
            mf_out.write({dataset_name : prop_data}, elements_per_file, output_filenames, "r+",
                         attrs={dataset_name : halo_lightcone_data[prop_name].attrs},
                         gzip=6, shuffle=True)

    
if __name__ == "__main__":

    # Get command line arguments
    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(description='Create lightcone halo catalogues.', comm=comm)
    parser.add_argument('lightcone_dir',  help='Directory with lightcone particle outputs')
    parser.add_argument('lightcone_base', help='Base name of the lightcone to use')
    parser.add_argument('halo_lightcone_filenames', help='Format string to generate halo lightcone filenames')
    parser.add_argument('soap_filenames', help='Format string to generate SOAP filenames')
    parser.add_argument('output_dir',     help='Where to write the output')
    parser.add_argument('--soap_so_name', type=str, default="SO/200_crit",
                        help='Name of SOAP group with the halo mass and radius, e.g. "SO/200_crit"')
    parser.add_argument('--overlap-method', type=str, default="fractional_radius", choices=list(overlap_methods),
                        help="How to assign particles which are in overlapping halos")
    args = parser.parse_args()

    message(f"Starting on {comm_size} MPI ranks")
    main(args)
    message("Done.")
