#!/bin/env python

import os
import sys
import numpy as np
import h5py
import argparse
from mpi4py import MPI
import unyt

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from virgo.mpi.util import MPIArgumentParser
from virgo.util.partial_formatter import PartialFormatter
import virgo.formats.swift as swift

import lightcone_io.halo_catalogue as hc

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Special most bound black hole ID for halos with no black holes
NULL_BH_ID = 0


def message(m):
    if comm_rank == 0:
        print(m)


def distributed_amax(arr, comm):
    """
    Find maximum over distributed array arr, bearing in mind that some
    MPI ranks might have zero elements.

    Returns None if all ranks have no elements.
    """
    if len(arr) == 0:
        local_max = None
    else:
        local_max = np.amax(arr)
    all_max = comm.allgather(local_max)
    global_max = None
    for m in all_max:
        if m is not None:
            if global_max is None:
                global_max = m
            elif m > global_max:
                global_max = m
    return global_max


def choose_bh_tracer(halo_index, snap_nr, final_snap_nr, snapshot_format,
                     membership_format, membership_cache):
    """
    Find the ID of a suitable tracer particle for each subhalo
    Ideally we want to pick a black hole that exists at the next
    and previous snapshots.

    halo_index: halo index used in the membership files
    snap_nr: snapshot at which the subhalos exist
    final_snap_nr: final snapshot number in the simulation
    snapshot_format: format string for snapshot filenames
    membership_format: format string for group membership filenames
    membership_cache: stores previously read BH halo membership
    """

    # Ensure we have the BH IDs and halo membership for snapshots snap_nr-1,
    # snap_nr and snap_nr+1 (may be in membership_cache already)
    for sn in (snap_nr+1, snap_nr, snap_nr-1):
        if (sn >= 0) and (sn <= final_snap_nr) and (sn not in membership_cache):

            # Discard excess cache entries
            while len(membership_cache) > 2:
                max_snap_in_cache = max(membership_cache.keys())
                del membership_cache[max_snap_in_cache]

            # Create unit registry for this snapshot
            if comm_rank == 0:
                with h5py.File(snapshot_format.format(snap_nr=sn, file_nr=0), "r") as infile:
                    reg = swift.soap_unit_registry_from_snapshot(infile)
            else:
                reg = None
            reg = comm.bcast(reg)

            # Read in the black hole particle IDs and positions for this snapshot
            filenames = PartialFormatter().format(snapshot_format, snap_nr=sn, file_nr=None)            
            mf1 = phdf5.MultiFile(filenames, file_nr_attr=("Header","NumFilesPerSnapshot"), comm=comm)
            snap_bh_ids, snap_bh_pos = mf1.read(("PartType5/ParticleIDs", "PartType5/Coordinates"), unpack=True, read_attributes=True)

            # Add unit info to the positions
            pos_units = swift.soap_units_from_attributes(snap_bh_pos.attrs, reg)
            snap_bh_pos = unyt.unyt_array(snap_bh_pos, units=pos_units)
            
            # Check for the case where there are no BHs at this snapshot: MultiFile.read()
            # returns None if no ranks read any elements.
            if snap_bh_ids is None:
                membership_cache[sn] = None
                continue

            # Read in the black hole particle halo membership
            filenames = PartialFormatter().format(membership_format, snap_nr=sn, file_nr=None)
            mf2 = phdf5.MultiFile(filenames, file_idx=mf1.all_file_indexes, comm=comm)
            (snap_bh_grnr, snap_bh_rank) = mf2.read(("PartType5/GroupNr_bound",
                                                     "PartType5/Rank_bound"), unpack=True)
            assert len(snap_bh_grnr) == len(snap_bh_ids)
            assert len(snap_bh_rank) == len(snap_bh_ids)

            # Add this snapshot to the cache
            membership_cache[sn] = (snap_bh_ids, snap_bh_grnr, snap_bh_rank, snap_bh_pos)
            nr_bh_local = len(snap_bh_ids)
            nr_bh_tot = comm.allreduce(nr_bh_local)
            message(f"    Read {nr_bh_tot} BHs for snapshot {sn}")

    # Check if we have any black holes at this snapshot
    if membership_cache[sn] is None:
        tracer_bh_id = np.ndarray(len(subhalo_id), dtype=np.int64)
        tracer_bh_id[:] = NULL_BH_ID
        tracer_bh_pos = -np.ones((len(subhalo_id),3), dtype=float)
        return tracer_bh_id, tracer_bh_pos

    # Get number of black holes at snapshot snap_nr
    nr_bh_local = len(membership_cache[snap_nr][0])
    nr_bh_tot = comm.allreduce(nr_bh_local)
    
    # Determine which black hole particles exist at the next snapshot
    if snap_nr < final_snap_nr:
        bh_id_this = membership_cache[snap_nr][0]
        bh_id_next = membership_cache[snap_nr+1][0]
        idx_at_next_snap = psort.parallel_match(bh_id_this, bh_id_next, comm=comm)
        exists_at_next_snap = idx_at_next_snap >= 0
    else:
        exists_at_next_snap = np.ones(nr_bh_local, dtype=bool)
    nr_existing_next = comm.allreduce(np.sum(exists_at_next_snap))
    message(f"    Number of BHs which exist at the next snapshot = {nr_existing_next}")

    # Determine which black hole particles exist at the previous snapshot
    if snap_nr > 0:
        bh_id_this = membership_cache[snap_nr][0]
        bh_id_prev = membership_cache[snap_nr-1][0]
        idx_at_prev_snap = psort.parallel_match(bh_id_this, bh_id_prev, comm=comm)
        exists_at_prev_snap = idx_at_prev_snap >= 0
    else:
        exists_at_prev_snap = np.ones(nr_bh_local, dtype=bool)
    nr_existing_prev = comm.allreduce(np.sum(exists_at_prev_snap))
    message(f"    Number of BHs which exist at the previous snapshot = {nr_existing_prev}")

    # Assign priorities to the BH particles. In descending order of importance:
    # 1. Should exist at the next timestep
    # 2. Should exist at the previous timestep
    # 3. Should be tightly bound
    rank_at_this_snap = membership_cache[snap_nr][2]
    max_rank = distributed_amax(rank_at_this_snap, comm)
    if max_rank is None:
        max_rank = 1
    bh_priority = (max_rank - rank_at_this_snap).astype(np.int64) # Low rank = high priority
    assert np.all(bh_priority>=0)
    bh_priority += (max_rank+1)*exists_at_prev_snap               # Boost priority if exists at snap_nr-1
    bh_priority += 2*(max_rank+1)*exists_at_next_snap             # Boost priority more if exists at snap_nr+1
    bh_id   = membership_cache[snap_nr][0]
    bh_grnr = membership_cache[snap_nr][1]
    bh_pos = membership_cache[snap_nr][3]
    message("    BH priorities assigned")

    # Discard BHs which are not in halos
    keep = bh_grnr >= 0
    bh_priority = bh_priority[keep]
    bh_id = bh_id[keep]
    bh_grnr = bh_grnr[keep]
    bh_pos = bh_pos[keep,:]

    # Sort BH particles by halo and then by priority within a halo
    max_priority = distributed_amax(bh_priority, comm)
    if max_priority is None:
        max_priority = 1
    sort_key = bh_grnr * (max_priority+1) + bh_priority
    order = psort.parallel_sort(sort_key, return_index=True, comm=comm)
    del sort_key
    bh_priority = psort.fetch_elements(bh_priority, order, comm=comm)
    bh_id = psort.fetch_elements(bh_id, order, comm=comm)
    bh_grnr =  psort.fetch_elements(bh_grnr, order, comm=comm)
    bh_pos = psort.fetch_elements(bh_pos, order, comm=comm)
    del order
    message("    Sorted BH particles by priority")

    # Now we need to discard all but the last (i.e. highest priority) particle in each halo.
    # Discard any particle which is in the same halo as the next particle on the same rank.
    keep = np.ones(len(bh_id), dtype=bool)
    keep[:-1] = bh_grnr[:-1] != bh_grnr[1:]
    
    # The last particle on each rank needs special treatment:
    # We need to know the group index of the first particle on the next rank
    # which has particles. Find the first group number on every rank.
    if len(bh_grnr) > 0:
        first_grnr = bh_grnr[0]
    else:
        first_grnr = None
    first_grnr = comm.allgather(first_grnr)
    
    # Ranks other than the last need to determine if their last particle
    # is in the same group as the first particle on a later rank.
    if comm_rank < (comm_size-1) and len(bh_grnr) > 0:
        first_grnr_next_rank = None
        for i in range(comm_rank+1, comm_size):
            if first_grnr[i] is not None:
                first_grnr_next_rank = first_grnr[i]
                break
        if first_grnr_next_rank is not None:
            if bh_grnr[-1] == first_grnr_next_rank:
                # A later rank has a particle in the same halo as our last particle.
                # The later particle will have a higher priority so discard this one.
                keep[-1] = False

    # Discard particles which are not the highest priority in their halo
    bh_id = bh_id[keep]
    bh_grnr = bh_grnr[keep]
    bh_pos = bh_pos[keep,:]
    del bh_priority
    message("    Discarded low priority BH particles")

    # Now that we only have the highest priority particle in each halo,
    # for each halo find the 0-1 BH particles which belong to that halo
    # and look up their IDs.
    ptr = psort.parallel_match(halo_index, bh_grnr, comm=comm)
    nr_groups_matched = comm.allreduce(np.sum(ptr>=0))
    nr_groups_total = comm.allreduce(len(ptr))
    fraction_matched = nr_groups_matched / nr_groups_total
    message(f"    Matched fraction {fraction_matched:.2f} of halos to BHs")

    # Fetch the IDs and positions of the matched black holes. Return ID=NULL_BH_ID where there's no match.
    tracer_bh_id = np.ndarray(len(halo_index), dtype=bh_id.dtype)
    tracer_bh_id[:] = NULL_BH_ID
    tracer_bh_id[ptr>=0] = psort.fetch_elements(bh_id, ptr[ptr>=0], comm=comm)
    tracer_bh_pos = np.zeros((len(halo_index),3), dtype=bh_pos.dtype) * bh_pos.units
    tracer_bh_pos[ptr>=0,:] = psort.fetch_elements(bh_pos, ptr[ptr>=0], comm=comm)

    # As a consistency check, fetch the group number of the matched particles:
    # Each matched particle should belong to the group it was matched to.
    tracer_bh_grnr = np.ndarray(len(halo_index), dtype=bh_grnr.dtype)
    tracer_bh_grnr[:] = -1
    tracer_bh_grnr[ptr>=0] = psort.fetch_elements(bh_grnr, ptr[ptr>=0], comm=comm)
    assert np.all(tracer_bh_grnr[ptr>=0] == halo_index[ptr>=0])
    
    return tracer_bh_id, tracer_bh_pos


def test_choose_bh_tracers():
    """
    Identify BH particles to use to place halos on the lightcone
    """
    parser = MPIArgumentParser(comm, description='Determine tracer particle for each halo.')
    parser.add_argument('halo_format', help='Format string for halo catalogue files')
    parser.add_argument('snapshot_format', help='Format string for snapshot filenames')
    parser.add_argument('membership_format', help='Format string for group membership filenames')
    parser.add_argument('first_snap', type=int, help='Index of the first snapshot to use')
    parser.add_argument('last_snap', type=int, help='Index of the last snapshot to use')
    parser.add_argument('snap_nr', type=int, help='Snapshot number to process')
    parser.add_argument('output_file', help='Where to write the output')
    args = parser.parse_args()

    to_read = ("InputHalos/cofp", "InputHalos/index")
    
    message("Reading halo catalogue")
    halo_cat = hc.HaloCatalogue(args.halo_format, args.first_snap, args.last_snap)
    halo_data = halo_cat.read(args.snap_nr, to_read)

    message("Finding tracers")
    membership_cache = {}
    tracer_id, tracer_pos = choose_bh_tracer(halo_data["InputHalos/index"], args.snap_nr, args.last_snap,
                                             args.snapshot_format, args.membership_format, membership_cache)
    
    message("Writing results")
    with h5py.File(args.output_file, "w", driver="mpio", comm=comm) as f:
        phdf5.collective_write(f, "tracer_id", tracer_id, comm)

    comm.barrier()
    message("Done.")


if __name__ == "__main__":
    test_choose_bh_tracers()
