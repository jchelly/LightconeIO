#!/bin/env python

import os
import sys
import time
t0 = time.time()

import numpy as np
import h5py
import argparse
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


def read_lightcone_halo_positions_and_radii(args):
    """
    Read in the lightcone halo catalogue and cross reference with SOAP
    to find the SO radius for each halo in the lightcone.
    """

    # Parallel read the halo catalogue: need (x,y,z), snapnum, id
    message("Reading lightcone halo catalogue")
    halo_lightcone_datasets = ("Xcminpot", "Ycminpot", "Zcminpot", "SnapNum", "ID")
    mf = phdf5.MultiFile(args.halo_lightcone_filenames, file_nr_attr=("Header", "NumberOfFiles"), comm=comm)
    halo_lightcone_data = mf.read(halo_lightcone_datasets, group="Subhalo")

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

    # The input catalogue is ordered by redshift, but we want a mix of redshifts on each rank
    message("Reassign halos to MPI ranks")
    nr_local_halos = len(halo_lightcone_data["ID"])
    rng = np.random.default_rng()
    sort_key = rng.integers(comm_size, size=nr_local_halos, dtype=np.int32)
    order = psort.parallel_sort(sort_key, comm=comm, return_index=True)
    for name in sorted(halo_lightcone_data):
        halo_lightcone_data[name] = psort.fetch_elements(halo_lightcone_data[name], order, comm=comm)

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
    radius_name = "SO/200_crit/SORadius"
    nr_halos = len(halo_lightcone_data["ID"])
    halo_lightcone_data[radius_name] = -np.ones(nr_halos, dtype=float)

    # Loop over snapshots
    for snapnum in unique_snap_all:

        # Datasets to read from SOAP
        soap_datasets = ("VR/ID", radius_name)

        # Read the SOAP catalogue for this snapshot
        message(f"Reading SOAP output for snapshot {snapnum}")
        mf = phdf5.MultiFile(args.soap_filenames % {"snap_nr" : snapnum}, file_idx=(0,), comm=comm)
        soap_data = mf.read(soap_datasets)

        # Match lightcone halos at this snapshot to SOAP halos by ID
        message("Finding lightcone halos in SOAP output")
        i1 = snap_offset_all[snapnum-min_snap]
        i2 = snap_offset_all[snapnum-min_snap] + snap_count_all[snapnum-min_snap]
        assert np.all(halo_lightcone_data["SnapNum"][i1:i2] == snapnum)
        ptr = psort.parallel_match(halo_lightcone_data["ID"][i1:i2], soap_data["VR/ID"], comm=comm)
        assert np.all(ptr>=0) # All halos in the lightcone should be found in SOAP

        # Store the SO radius for each lightcone halo
        message("Storing SO radii for lightcone halos at this snapshot")
        halo_lightcone_data[radius_name][i1:i2] = psort.fetch_elements(soap_data[radius_name], ptr, comm=comm)

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
    for filename in local_particle_files:
        with h5py.File(filename, "r") as infile:
            pos.append(infile[ptype]["Coordinates"][...])
    if len(pos) > 0:
        pos = np.concatenate(pos)
    else:
        pos = None
    pos = mpi_util.replace_none_with_zero_size(pos, comm=comm)

    nr_particles_local = pos.shape[0]
    nr_particles_total = comm.allreduce(nr_particles_local)
    message(f"Read in {nr_particles_total} particles")

    return pos


    # For each particle type
    #  - parallel read lightcone particles
    #  - build kdtree for local particles
    #  - tree search and tag particles within r200c of local halos with halo ID (what to do about overlap?)
    #  - pass halos to next MPI task and repeat

    # Write new file with particle halo membership
    
    # Possible optimization:
    # Partition particles by x coord and send each rank just a slice of the halo catalogue


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
    #halo_lightcone_data = read_lightcone_halo_positions_and_radii(args)

    # Locate the particle data
    type_z_range, all_particle_files = read_lightcone_index(args)

    # Loop over types to do
    for ptype in ("BH",):
        
        # Read in positions of lightcone particles of this type
        pos = read_lightcone_particles(all_particle_files, ptype)



    comm.barrier()
    message("Done.")