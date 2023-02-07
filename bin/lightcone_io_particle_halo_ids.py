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

class ArgumentParserError(Exception): pass

class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(message+"\n")
        raise ArgumentParserError(message)


def message(m):
    if comm_rank == 0:
        t1 = time.time()
        elapsed = t1-t0
        print(f"{elapsed:.1f}s: {m}")


def read_lightcone_halo_positions_and_radii(args, radius_name):
    """
    Read in the lightcone halo catalogue and cross reference with SOAP
    to find the SO radius for each halo in the lightcone.

    Assumes that positions in the lightcone are comoving and in the
    same units as SOAP (except for the expansion factor dependence).
    """

    # Parallel read the halo catalogue: need (x,y,z), snapnum, id
    message("Reading lightcone halo catalogue")
    halo_lightcone_datasets = ("Xcminpot", "Ycminpot", "Zcminpot", "SnapNum", "ID")
    mf = phdf5.MultiFile(args.halo_lightcone_filenames, file_nr_attr=("Header", "NumberOfFiles"), comm=comm)
    halo_lightcone_data = mf.read(halo_lightcone_datasets, group="Subhalo", read_attributes=True)

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
    halo_pos = np.column_stack((halo_lightcone_data["Xcminpot"],
                                halo_lightcone_data["Ycminpot"],
                                halo_lightcone_data["Zcminpot"]))
    del halo_lightcone_data["Xcminpot"]
    del halo_lightcone_data["Ycminpot"]
    del halo_lightcone_data["Zcminpot"]
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
    unique_snap, snap_offset, snap_count = np.unique(halo_lightcone_data["SnapNum"], return_index=True, return_counts=True)

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
    halo_lightcone_data[radius_name] = -np.ones(nr_halos, dtype=float)

    # Loop over snapshots
    for snapnum in unique_snap_all:

        # Datasets to read from SOAP
        soap_datasets = ("VR/ID", radius_name)

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

        # Store the SO radius for each lightcone halo
        message("Storing SO radii for lightcone halos at this snapshot")
        psort.fetch_elements(soap_data[radius_name], ptr, result=halo_lightcone_data[radius_name][i1:i2], comm=comm)

    # All halos should have been assigned a radius
    assert np.all(halo_lightcone_data[radius_name] >= 0)

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


def read_lightcone_particles(all_particle_files, ptype):
    """
    Read in the lightcone particle positions for one particle type
    """

    nr_files = len(all_particle_files)
    message(f"Reading positions for type {ptype} from {nr_files} files")

    # Assign files to MPI ranks and find files this rank will read
    files_per_rank = np.zeros(comm_size, dtype=int)
    files_per_rank[:] = len(all_particle_files) // comm_size
    files_per_rank[:len(all_particle_files) % comm_size] += 1
    assert np.sum(files_per_rank) == len(all_particle_files)
    first_file_rank = np.cumsum(files_per_rank) - files_per_rank
    local_particle_files = all_particle_files[first_file_rank[comm_rank]:first_file_rank[comm_rank]+files_per_rank[comm_rank]]
    
    # Read the positions for this particle type
    pos = []
    particles_per_file = []
    for filename in local_particle_files:
        with h5py.File(filename, "r") as infile:
            pos.append(infile[ptype]["Coordinates"][...])
            particles_per_file.append(pos[-1].shape[0])
    if len(pos) > 0:
        pos = np.concatenate(pos)
    else:
        pos = None
    pos = mpi_util.replace_none_with_zero_size(pos, comm=comm)
    particles_per_file = np.concatenate(comm.allgather(particles_per_file))
    nr_particles_local = pos.shape[0]
    nr_particles_total = comm.allreduce(nr_particles_local)
    message(f"Read in {nr_particles_total} particles")

    return files_per_rank, first_file_rank, particles_per_file, pos


def compute_particle_group_index(halo_id, halo_pos, halo_radius, part_pos):
    """
    Tag particles which are within the SO radius of a halo
    """

    # Assign indexes to the particles so we can restore their ordering later
    nr_particles = part_pos.shape[0]
    offset = comm.scan(nr_particles) - nr_particles
    part_index = np.arange(nr_particles, dtype=np.int64) + offset

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

    # Determine the range of x coordinates of halos which could overlap particles on this rank
    local_x_min = np.amin(part_pos[:,0]) - max_radius
    x_min_on_rank = np.asarray(comm.allgather(local_x_min), dtype=part_pos.dtype)
    local_x_max = np.amax(part_pos[:,0]) + max_radius
    x_max_on_rank = np.asarray(comm.allgather(local_x_min), dtype=part_pos.dtype)

    # Sort local halos by x coordinate
    message("Sorting local lightcone halos by x coordinate")
    order = np.argsort(halo_pos[:,0])
    halo_pos[...] = halo_pos[order,:]
    halo_id[...] = halo_id[order]

    # Determine what range of halos needs to be sent to each MPI rank
    first_halo_for_rank = np.searchsorted(halo_pos[:,0], x_min_on_rank, side="left")
    last_halo_for_rank = np.searchsorted(halo_pos[:,0], x_max_on_rank, side="right")
    nr_halos_for_rank = last_halo_for_rank - first_halo_for_rank

    # Compute lengths and offsets for alltoallv halo exchange
    send_offset = first_halo_for_rank
    send_count = nr_halos_for_rank
    recv_count = comm.alltoall(send_count)
    recv_offset = np.cumsum(recv_count) - recv_count

    # Exchange halo IDs
    message("Exchanging halo IDs")
    halo_id_recv = np.empty_like(halo_id)
    my_alltoallv(halo_id, send_count, send_offset,
                 halo_id_recv, recv_count, recv_offset,
                 comm=comm)
    halo_id = halo_id_recv
    del halo_id_recv

    # Exchange halo positions
    message("Exchanging halo positions")
    halo_pos_recv = np.empty_like(halo_pos)
    my_alltoallv(halo_pos.flatten(), send_count*3, send_offset*3,
                 halo_pos_recv.flatten(), recv_count*3, recv_offset*3,
                 comm=comm)
    halo_pos = halo_pos_recv
    del halo_pos_recv

    # Allocate output array for the particle halo IDs
    nr_parts = part_pos.shape[0]
    part_halo_id = -np.ones(nr_parts, dtype=halo_id.dtype)

    # Build a kdtree with the local particles
    message("Building kdtree")
    tree = scipy.spatial.KDTree(part_pos)
        
    # Loop over local halos
    message("Assigning halo IDs to particles")
    for i in range(len(halo_id)):
            
        # Identify particles within this halo's radius
        idx = tree.query_ball_point(halo_pos[i,:], halo_radius[i])

        # Tag particles within the radius with the halo ID
        part_halo_id[idx] = halo_id[i]

    # Tidy up
    del halo_id
    del halo_pos
    del part_pos

    # Restore original particle ordering and return halo IDs
    message("Restoring particle order")
    order = psort.parallel_sort(part_index, return_index=True, comm=comm)
    del part_index
    psort.fetch_elements(part_halo_id, order, result=part_halo_id, comm=comm)

    return part_halo_id


if __name__ == "__main__":

    # Get command line arguments
    if comm.Get_rank() == 0:
        os.environ['COLUMNS'] = '80' # Can't detect terminal width when running under MPI?
        parser = ThrowingArgumentParser(description='Create lightcone halo catalogues.')
        parser.add_argument('lightcone_dir',  help='Directory with lightcone particle outputs')
        parser.add_argument('lightcone_base', help='Base name of the lightcone to use')
        parser.add_argument('halo_lightcone_filenames', help='Format string to generate halo lightcone filenames')
        parser.add_argument('soap_filenames', help='Format string to generate SOAP filenames')
        parser.add_argument('output_dir',     help='Where to write the output')
        try:
            args = parser.parse_args()
        except ArgumentParserError as e:
            args = None
    else:
        args = None
    args = comm.bcast(args)
    if args is None:
        MPI.Finalize()
        sys.exit(0)

    message(f"Starting on {comm_size} MPI ranks")

    # Read in position and radius for halos in the lightcone
    radius_name = "SO/200_crit/SORadius"
    halo_lightcone_data = read_lightcone_halo_positions_and_radii(args, radius_name)

    # Locate the particle data
    type_z_range, all_particle_files = read_lightcone_index(args)

    # Loop over types to do
    create_files = True
    for ptype in ("BH",):
        
        # Read in positions of lightcone particles of this type
        message(f"Reading particles of type {ptype}")
        (files_per_rank, first_file_rank,
         particles_per_file, part_pos) = read_lightcone_particles(all_particle_files, ptype)

        # Assign group indexes to the particles
        message("Assigning group indexes")
        halo_id = halo_lightcone_data["ID"]
        halo_pos = halo_lightcone_data["Pos_minpot"]
        halo_radius = halo_lightcone_data[radius_name]
        part_halo_id = compute_particle_group_index(halo_id, halo_pos, halo_radius, part_pos)

        # Write out the particle halo IDs
        # Loop over files to write on this rank
        for file_nr in range(first_file_rank[comm_rank],
                             first_file_rank[comm_rank]+files_per_rank[comm_rank]):
            
            # Find range of particles to write to this file
            offset = np.sum(particles_per_file[first_file_rank[comm_rank]:file_nr], dtype=int)
            length = particles_per_file[file_nr]
            print(f"Rank {comm_rank} is writing {length} particles to file {file_nr}")

            # Open or create the output file
            filename = f"{output_dir}/particle_halo_ids.{file_nr}.hdf5"
            mode = "w" if create_file else "r+"
            outfile = h5py.File(outfile, mode)

            # Write out the particle halo ID data
            group = outfile.create_group(ptype)
            dataset = group.create_dataset("HaloID", data=part_halo_id[offset:offset+length],
                                           compression="gzip", compression_opts=6)
            outfile.close()
        
        # Tidy up before reading next particle type
        del part_pos
        del part_halo_id

        # Only need to create new output files for the first type
        create_files = False
        
    comm.barrier()
    message("Done.")
