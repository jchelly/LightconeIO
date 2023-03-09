#!/bin/env python

import os
import sys
import numpy as np
import h5py
import argparse
import collections
from mpi4py import MPI

import virgo.util.match as match
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from virgo.mpi.util import MPIArgumentParser
import lightcone_io.particle_reader as pr

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Special most bound black hole ID for halos with no black holes
NULL_BH_ID = 0


def message(m):
    if comm_rank == 0:
        print(m)


def choose_bh_tracer(subhalo_id, snap_nr, final_snap_nr, snapshot_format,
                     membership_format, membership_cache):
    """
    Find the ID of a suitable tracer particle for each subhalo
    Ideally we want to pick a black hole that exists at the next
    and previous snapshots.

    subhalo_id: array of subhalo IDs at snapshot snap_nr
    snap_nr: snapshot at which the subhalos exist
    final_snap_nr: final snapshot number in the simulation
    snapshot_format: format string for snapshot filenames
    membership_format: format string for group membership filenames
    membership_cache: stores previously read BH halo membership
    """

    # Ensure we have the BH IDs and halo membership for snapshots snap_nr-1,
    # snap_nr and snap_nr+1 (may be in membership_cache already)
    for sn in (snap_nr-1, snap_nr, snap_nr+1):
        if (sn >= 0) and (sn <= final_snap_nr) and (sn not in membership_cache):

            # Discard excess cache entries (assumes cache is ordered dict)
            while len(membership_cache) > 2:
                membership_cache.popitem(last=False)

            # Read in the black hole particle IDs for this snapshot
            mf1 = phdf5.MultiFile(snapshot_format, file_nr_attr=("Header","NumFilesPerSnapshot"), comm=comm)
            snap_bh_ids = mf1.read("PartType5/ParticleIDs")

            # Read in the black hole particle halo membership
            mf2 = phdf5.MultiFile(membership_format, file_idx=mf1.all_file_indexes, comm=comm)
            data = mf2.read(("PartType5/GroupNr_bound", "PartType5/Rank_bound"))
            snap_bh_grnr = data["PartType5/GroupNr_bound"]
            snap_bh_rank = data["PartType5/Rank_bound"]
            assert len(snap_bh_grnr) == len(snap_bh_ids)
            assert len(snap_bh_rank) == len(snap_bh_ids)

            # Add this snapshot to the cache
            membership_cache[sn] = (snap_bh_ids, snap_bh_grnr, snap_bh_rank)
            nr_bh_local = len(snap_bh_ids)
            nr_bh_tot = comm.allreduce(nr_bh_local)
            message(f"Read {nr_bh_tot} BHs for snapshot {sn}")

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
    message(f"Number of BHs which exist at the next snapshot = {nr_existing_next}")

    # Determine which black hole particles exist at the previous snapshot
    if snap_nr > 0:
        bh_id_this = membership_cache[snap_nr][0]
        bh_id_prev = membership_cache[snap_nr-1][0]
        idx_at_prev_snap = psort.parallel_match(bh_id_this, bh_id_prev, comm=comm)
        exists_at_prev_snap = idx_at_prev_snap >= 0
    else:
        exists_at_prev_snap = np.ones(nr_bh_local, dtype=bool)
    nr_existing_prev = comm.allreduce(np.sum(exists_at_prev_snap))
    message(f"Number of BHs which exist at the previous snapshot = {nr_existing_prev}")

    # Assign priorities to the BH particles. In descending order of importance:
    # 1. Should exist at the next timestep
    # 2. Should exist at the previous timestep
    # 3. Should be tightly bound
    rank_at_this_snap = membership_cache[snap_nr][2]
    max_rank = comm.allreduce(np.amax(rank_at_this_snap))
    bh_priority = (max_rank - rank_at_this_snap).astype(np.int64) # Low rank = high priority
    bh_priority += (max_rank+1)*exists_at_prev_snap               # Boost priority if exists at snap_nr-1
    bh_priority += 2*(max_rank+1)*exists_at_next_snap             # Boost priority more if exists at snap_nr+1
    bh_id   = membership_cache[snap_nr][0]
    bh_grnr = membership_cache[snap_nr][1]
    message("BH priorities assigned")

    # Discard BHs which are not in halos
    keep = bh_grnr >= 0
    bh_priority = bh_priority[keep]
    bh_id = bh_id[keep]
    bh_grnr = bh_grnr[keep]

    # Sort BH particles by halo and then by priority within a halo
    sort_key = bh_grnr * (np.amax(bh_priority)+1) + bh_priority
    order = psort.parallel_sort(sort_key, return_index=True, comm=comm)
    del sort_key
    bh_priority = psort.fetch_elements(bh_priority, order, comm=comm)
    bh_id = psort.fetch_elements(bh_id, order, comm=comm)
    bh_grnr =  psort.fetch_elements(bh_grnr, order, comm=comm)
    del order
    message("Sorted BH particles by priority")

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
    del bh_priority
    message("Discarded low priority BH particles")

    # Now that we only have the highest priority particle in each halo,
    # for each VR halo find the 0-1 BH particles which belong to that halo
    # and look up their IDs. Here was assume that a VR group's array index
    # in the catalogue is equal to it's ID minus 1.
    ptr = psort.parallel_match(subhalo_id-1, bh_grnr, comm=comm)
    nr_groups_matched = comm.allreduce(np.sum(ptr>=0))
    nr_groups_total = comm.allreduce(len(ptr))
    fraction_matched = nr_groups_matched / nr_groups_total
    message(f"Matched fraction {fraction_matched} of VR groups to BHs")

    # Fetch the IDs of the matched black holes. Return NULL_BH_ID where there's no match.
    tracer_bh_id = np.ndarray(len(subhalo_id), dtype=bh_id.dtype)
    tracer_bh_id[:] = NULL_BH_ID
    tracer_bh_id[ptr>=0] = psort.fetch_elements(bh_id, ptr[ptr>=0], comm=comm)

    # As a consistency check, fetch the group number of the matched particles:
    # Each matched particle should belong to the VR group it was matched to.
    tracer_bh_grnr = np.ndarray(len(subhalo_id), dtype=bh_grnr.dtype)
    tracer_bh_grnr[:] = -1
    tracer_bh_grnr[ptr>=0] = psort.fetch_elements(bh_grnr, ptr[ptr>=0], comm=comm)
    assert np.all(tracer_bh_grnr[ptr>=0] == subhalo_id[ptr>=0]-1)

    return tracer_bh_id


def match_black_holes(args):

    message(f"Starting on {comm_size} MPI ranks")

    # Open the lightcone particle output in MPI mode
    filename = f"{args.lightcone_dir}/{args.lightcone_base}_particles/{args.lightcone_base}_0000.0.hdf5"
    lightcone = pr.IndexedLightcone(filename, comm=comm)

    # Get snapshot redshifts and other metadata from the tree file
    if comm_rank == 0:
        with h5py.File(args.tree_filename % {"file_nr" : 0}, "r") as treefile:
            # Read snapshot redshifts
            output_snapshots = treefile["Snapshots/SnapNum"][...]
            output_redshifts = treefile["Snapshots/Redshift"][...]
            # Also read in VR unit information
            vr_unit_info = {}
            for name in treefile["UnitInfo"].attrs:
                vr_unit_info[name] = float(treefile["UnitInfo"].attrs[name].decode())
            # And simulation information
            vr_sim_info = {}
            for name in treefile["SimulationInfo"].attrs:
                vr_sim_info[name] = treefile["SimulationInfo"].attrs[name]
        max_snap_nr = np.amax(output_snapshots)
        redshifts = -np.ones(max_snap_nr+1, dtype=float)
        for outs, outr in zip(output_snapshots, output_redshifts):
            redshifts[outs] = outr
    else:
        redshifts = None
        vr_unit_info = None
        vr_sim_info = None
    redshifts, vr_unit_info, vr_sim_info = comm.bcast((redshifts, vr_unit_info, vr_sim_info))
    final_snap = len(redshifts) - 1

    # Get physical constants etc from SWIFT:
    # These are needed to interpret VR unit metadata since we want to
    # assume exactly the same definitions of Mpc, Msolar etc that SWIFT used.
    if comm_rank == 0:
        physical_constants_cgs = {}
        snapshot_units = {}
        filename = args.snapshot_format.format(snap_nr=final_snap, file_nr=0)
        with h5py.File(filename, "r") as infile:
            group = infile["PhysicalConstants/CGS"]
            for name in group.attrs:
                physical_constants_cgs[name] = float(group.attrs[name])
            group = infile["Units"]
            for name in group.attrs:
                snapshot_units[name] = float(group.attrs[name])
            boxsize = infile["Header"].attrs["BoxSize"]
            assert np.all(boxsize==boxsize[0])
            boxsize = boxsize[0]
    else:
        physical_constants_cgs = None
        snapshot_units = None
        boxsize = None
    physical_constants_cgs, snapshot_units, boxsize = comm.bcast((physical_constants_cgs, snapshot_units, boxsize))

    # Read the merger tree data we need:
    # The Redshift and [XYZ]cminpot arrays will be updated to the point of lightcone crossing.
    position_props = ("Subhalo/Xcmbp_bh", "Subhalo/Ycmbp_bh", "Subhalo/Zcmbp_bh",
                      "Subhalo/Xcminpot", "Subhalo/Ycminpot", "Subhalo/Zcminpot")
    mass_props     = ("Subhalo/Mass_tot", "Subhalo/Mass_star", "Subhalo/Mass_gas", "Subhalo/Mass_bh")
    other_props    = ("Subhalo/ID_mbp_bh", "Subhalo/n_bh", "Subhalo/Structuretype", "Subhalo/SnapNum", "Subhalo/ID")
    merger_tree_props = (position_props + mass_props + other_props)
    treefile = phdf5.MultiFile(args.tree_filename,
                               file_nr_attr=("Header", "NumberOfFiles"),
                               comm=comm)
    merger_tree = treefile.read(merger_tree_props)
    message("Read in merger trees")

    # Assign redshifts to the subhalos
    merger_tree["Subhalo/Redshift"] = redshifts[merger_tree["Subhalo/SnapNum"]]
    assert np.all(merger_tree["Subhalo/Redshift"] > -0.5)

    # Will not try to handle the (very unlikely) case where some ranks have zero halos
    nr_halos = len(merger_tree["Subhalo/Redshift"])
    assert nr_halos > 0

    # Determine redshifts in the merger tree
    redshifts = np.unique(merger_tree["Subhalo/Redshift"]) # Find unique values on this rank
    redshifts = np.concatenate(comm.allgather(redshifts))  # Combine arrays from different ranks
    redshifts = np.unique(redshifts)                       # Find unique values over all ranks
    message("Identified snapshot redshifts")
    for i, z in enumerate(redshifts):
        message(f"  {i} : {z}")
    
    # Sort local halos by redshift
    order = np.argsort(merger_tree["Subhalo/Redshift"])
    for name in merger_tree:
        merger_tree[name] = merger_tree[name][order,...]
    message("Sorted halos by redshift")

    # Find range of local halos at each redshift
    first_at_redshift = np.searchsorted(merger_tree["Subhalo/Redshift"], redshifts, side="left")
    first_at_next_redshift = np.searchsorted(merger_tree["Subhalo/Redshift"], redshifts, side="right")
    nr_at_redshift = first_at_next_redshift - first_at_redshift
    snap_nr = -np.ones(len(redshifts), dtype=int)
    for redshift_nr in range(len(redshifts)):
        z = merger_tree["Subhalo/Redshift"][first_at_redshift[redshift_nr]:first_at_next_redshift[redshift_nr]]
        if not(np.all(z==redshifts[redshift_nr])):
            raise RuntimeError("Redshift ranges not identified correctly!")
        snap_nr[redshift_nr] = merger_tree["Subhalo/SnapNum"][first_at_redshift[redshift_nr]]
    assert np.all(snap_nr >= 0)
    message("Identified range of halos at each redshift")

    # Special ID used to indicate no black hole. Will check that this doesn't appear as a real ID.
    no_bh = merger_tree["Subhalo/n_bh"]==0
    assert np.all(merger_tree["Subhalo/ID_mbp_bh"][no_bh] == NULL_BH_ID)

    # Loop over unique redshifts in the trees, excluding the last
    halos_so_far = 0
    membership_cache = collections.OrderedDict()
    for redshift_nr in range(len(redshifts)):
        
        # Find the range of halos which exist at this redshift
        i1 = first_at_redshift[redshift_nr]
        i2 = first_at_next_redshift[redshift_nr]
        nr_halos_in_slice = i2-i1
        nr_halos_in_slice_all = comm.allreduce(nr_halos_in_slice)

        # Choose the tracer particle to use for each object:
        # This overrides the most bouhnd BH from velociraptor.
        merger_tree["Subhalo/ID_mbp_bh"][i1:i2] = choose_bh_tracer(merger_tree["Subhalo/ID"][i1:i2],
                                                                   snap_nr, final_snap, args.snapshot_format,
                                                                   args.membership_format, membership_cache)

        # Each snapshot populates a redshift range which reaches half way to adjacent snapshots
        # (range is truncated for the first and last snapshots)
        if redshift_nr == 0:
            z1 = redshifts[redshift_nr]
        else:
            z1 = 0.5*(redshifts[redshift_nr-1]+redshifts[redshift_nr])
        if redshift_nr == len(redshifts)-1:
            z2 = redshifts[redshift_nr]
        else:
            z2 = 0.5*(redshifts[redshift_nr]+redshifts[redshift_nr+1])

        message(f"  Using {nr_halos_in_slice_all} halos at z={redshifts[redshift_nr]:.3f} to populate range z={z1:.3f} to z={z2:.3f}")

        # Find halo most bound BH IDs
        id_mbp_bh = merger_tree["Subhalo/ID_mbp_bh"][i1:i2]
        have_bh   = merger_tree["Subhalo/n_bh"][i1:i2] > 0
        mass      = merger_tree["Subhalo/Mass_tot"][i1:i2] * 1.0e10

        # Find fraction of halos with BHs as a function of mass
        log10_mmin = 8.0
        log10_mmax = 16.0
        nbins = 40
        bins = np.logspace(log10_mmin, log10_mmax, nbins+1)
        nr_halos, bin_edges = np.histogram(mass, bins=bins)
        nr_with_bh, bin_edges = np.histogram(mass[have_bh], bins=bins)
        nr_halos = comm.allreduce(nr_halos)
        nr_with_bh = comm.allreduce(nr_with_bh)
        frac_with_bh = np.divide(nr_with_bh, nr_halos, out=np.zeros_like(nr_halos, dtype=float), where=nr_halos>0)
        bin_centres = np.sqrt(bin_edges[1:]*bin_edges[:-1])

        # Read in the lightcone BH particle positions and IDs in this redshift range
        lightcone_props = ("Coordinates", "ParticleIDs", "ExpansionFactors")
        particle_data = lightcone["BH"].read_exact(lightcone_props, redshift_range=(z1,z2))
        if np.any(particle_data["ParticleIDs"] == NULL_BH_ID):
            raise RuntimeError("Found a BH particle with ID=NULL_BH_ID!")
        nr_parts = len(particle_data["ParticleIDs"])
        nr_parts_all = comm.allreduce(nr_parts)
        message(f"  Read in {nr_parts_all} lightcone BH particles for this redshift range")

        # Try to match BH particles to the halo most bound BH IDs.
        # There may be multiple particles matching each halo due to the periodicity of the box.
        # Since halos with no black hole have ID_mbp_bh=NULL_BH_ID and this value never appears
        # in the particle data, every match will become a halo in the output catalogue.
        halo_index = psort.parallel_match(particle_data["ParticleIDs"], id_mbp_bh, comm=comm)
        matched = halo_index>=0
        nr_matched = np.sum(matched)
        nr_matched_all = comm.allreduce(nr_matched)
        halos_so_far += nr_matched
        halo_index = halo_index[matched]
        pc_matched = 100.0*(nr_matched_all/nr_parts_all)
        message(f"  Matched {nr_matched_all} BH particles in this slice ({pc_matched:.2f}%)")

        # Create the output halo catalogue for this redshift slice:
        # For each matched BH particle in the lightcone, we fetch the properties of the halo
        # it was matched with.
        halo_slice = {}
        for name in merger_tree:
            halo_slice[name] = psort.fetch_elements(merger_tree[name][i1:i2,...], halo_index, comm=comm)
        message(f"  Found halo properties for this slice")

        # Find conversion factor to put positions into comoving, no h units.
        # This needs to be a at the redshift of the snapshot the halo is taken from.
        a = 1.0/(1.0+halo_slice["Subhalo/Redshift"])
        assert np.all(a[0]==a)
        a = float(a[0])
        h = float(vr_sim_info["h_val"])
        if vr_unit_info["Comoving_or_Physical"] == 0:
            # VR position units are physical with no h dependence. Need to convert to comoving.
            halo_pos_conversion = 1.0/a
            mass_h_exponent = 0.0
        else:
            # VR position units are comoving 1/h. Multiply out the h factor.
            halo_pos_conversion = 1.0/h
            mass_h_exponent = -1.0

        # Convert VR halo positions into SWIFT snapshot length units
        swift_length_unit_in_mpc = snapshot_units["Unit length in cgs (U_L)"] / (1.0e6*physical_constants_cgs["parsec"])
        halo_pos_conversion *= (vr_unit_info["Length_unit_to_kpc"]/1000.0) / swift_length_unit_in_mpc

        # Construct metadata for output quantites
        # Lengths: will be converted to comoving, no h, swift snapshot units
        length_unit_cgs = snapshot_units["Unit length in cgs (U_L)"]
        length_attrs = {
            "U_I exponent" : (0.0,),
            "U_L exponent" : (1.0,),
            "U_M exponent" : (0.0,),
            "U_T exponent" : (0.0,),
            "U_t exponent" : (0.0,),
            "a-scale exponent" : (1.0,),
            "h-scale exponent" : (0.0,),
            "Conversion factor to CGS (not including cosmological corrections)" : (length_unit_cgs,),
            "Conversion factor to CGS (including cosmological corrections)" : (a*length_unit_cgs,),
        }
        
        # Masses: just describe the existing VR units
        mass_unit_cgs = vr_unit_info["Mass_unit_to_solarmass"] * physical_constants_cgs["solar_mass"]
        mass_attrs = {
            "U_I exponent" : (0.0,),
            "U_L exponent" : (0.0,),
            "U_M exponent" : (1.0,),
            "U_T exponent" : (0.0,),
            "U_t exponent" : (0.0,),
            "a-scale exponent" : (0.0,),
            "h-scale exponent" : (mass_h_exponent,),
            "Conversion factor to CGS (not including cosmological corrections)" : (mass_unit_cgs,),
            "Conversion factor to CGS (including cosmological corrections)" : (mass_unit_cgs*(h**mass_h_exponent),),
        }
        
        # Dimensionless quantities
        dimensionless_attrs = {
            "U_I exponent" : (0.0,),
            "U_L exponent" : (0.0,),
            "U_M exponent" : (0.0,),
            "U_T exponent" : (0.0,),
            "U_t exponent" : (0.0,),
            "a-scale exponent" : (0.0,),
            "h-scale exponent" : (0.0,),
            "Conversion factor to CGS (not including cosmological corrections)" : (1.0,),
            "Conversion factor to CGS (including cosmological corrections)" : (1.0,),
        }

        # Compute the position of each halo in the output:
        #
        # We know:
        #  - the position in the lightcone of the most bound black hole particle
        #  - the position in the snapshot of the most bound black hole particle
        #  - the position in the snapshot of the halo's potential minimum
        #
        # We want to compute the position in the lightcone of the potential minimum.
        #

        # Position of the matched BH particle in the lightcone particle output
        bh_pos_in_lightcone = particle_data["Coordinates"][matched,...].ndarray_view()

        # Position of the halo's most bound black hole, from VR
        bh_pos_in_snapshot  = np.column_stack((halo_slice["Subhalo/Xcmbp_bh"],
                                               halo_slice["Subhalo/Ycmbp_bh"],
                                               halo_slice["Subhalo/Zcmbp_bh"])) * halo_pos_conversion
        
        # Position of the halo's potential minimum, from VR
        halo_pos_in_snapshot = np.column_stack((halo_slice["Subhalo/Xcminpot"],
                                                halo_slice["Subhalo/Ycminpot"],
                                                halo_slice["Subhalo/Zcminpot"])) * halo_pos_conversion
        
        # Vector from the most bound BH to the potential minimum - may need box wrapping
        bh_to_minpot_vector = halo_pos_in_snapshot - bh_pos_in_snapshot
        bh_to_minpot_vector = ((bh_to_minpot_vector+0.5*boxsize) % boxsize) - 0.5*boxsize

        # Compute halo potential minimum position in lightcone
        halo_pos_in_lightcone = bh_pos_in_lightcone + bh_to_minpot_vector

        # Report largest offset between BH and potential minimum
        max_offset = np.amax(np.abs(bh_to_minpot_vector.flatten()))
        max_offset = comm.allreduce(max_offset, op=MPI.MAX)
        if comm_rank == 0:
            message(f"  Maximum minpot/bh offset = {max_offset} (SWIFT length units, comoving)")

        # Overwrite the halo position in the output catalogue
        halo_slice["Subhalo/Xcminpot"] = halo_pos_in_lightcone[:,0]
        halo_slice["Subhalo/Ycminpot"] = halo_pos_in_lightcone[:,1]
        halo_slice["Subhalo/Zcminpot"] = halo_pos_in_lightcone[:,2]
        halo_slice["Subhalo/Xcmbp_bh"] = bh_pos_in_lightcone[:,0]
        halo_slice["Subhalo/Ycmbp_bh"] = bh_pos_in_lightcone[:,1]
        halo_slice["Subhalo/Zcmbp_bh"] = bh_pos_in_lightcone[:,2]
        # Set the redshift of each halo to the redshift of lightcone crossing
        halo_slice["Subhalo/Redshift"] = 1.0/particle_data["ExpansionFactors"][matched]-1.0
        message(f"  Computed potential minimum position in lightcone")

        # Function to add attributes to a dataset
        def write_attributes(dset, attrs):
            for name in attrs:
                dset.attrs[name] = attrs[name]

        # Write out the halo catalogue for this snapshot
        output_filename = f"{args.output_dir}/lightcone_halos_{redshift_nr:04d}.hdf5"
        outfile = h5py.File(output_filename, "w", driver="mpio", comm=comm)
        outfile.create_group("Subhalo")
        for name in halo_slice:
            # Write the data
            writebuf = np.ascontiguousarray(halo_slice[name])
            dset = phdf5.collective_write(outfile, name, writebuf, comm=comm)
            # Add unit info
            if name in position_props:
                write_attributes(dset, length_attrs)
            elif name in mass_props:
                write_attributes(dset, mass_attrs)
            else:
                write_attributes(dset, dimensionless_attrs)
        outfile.close()

        # Count halos output so far, including this redshift slice
        halos_so_far_all = comm.allreduce(halos_so_far)

        # Add the completeness information etc
        comm.barrier()
        if comm_rank == 0:
            outfile = h5py.File(output_filename, "r+")
            grp = outfile.create_group("Completeness")
            grp["MassBinCentre"] = bin_centres
            grp["NumberOfHalos"] = nr_halos
            grp["NumberOfHalosWithBH"] = nr_with_bh
            grp["FractionWithBH"] = frac_with_bh
            grp = outfile.create_group("Header")
            grp.attrs["MinimumRedshift"] = z1
            grp.attrs["MaximumRedshift"] = z2
            grp.attrs["AllRedshiftsWithHalos"] = redshifts
            grp.attrs["NumberOfFiles"] = len(redshifts)-1
            grp.attrs["ThisFile"] = redshift_nr
            grp.attrs["NumberOfHalosInFile"] = nr_matched_all
            grp.attrs["CumulativeNumberOfHalos"] = halos_so_far_all
            grp.attrs["LightconeDir"] = args.lightcone_dir
            grp.attrs["LightconeBase"] = args.lightcone_base
            grp.attrs["TreeFileName"] = args.tree_filename
            outfile.close()
        message(f"  Wrote file: {output_filename}")

        # Tidy up particle arrays before we read the next slice
        del particle_data

    comm.barrier()
    message("All redshift ranges done.")


def test_choose_bh_tracers():
    """
    Identify BH particles to use to place VR halos on the lightcone
    """
    parser = MPIArgumentParser(comm, description='Determine tracer particle for each VR halo.')
    parser.add_argument('vr_format', help='Format string for VR properties files')
    parser.add_argument('snapshot_format', help='Format string for snapshot filenames')
    parser.add_argument('membership_format', help='Format string for group membership filenames')
    parser.add_argument('snap_nr', type=int, help='Snapshot number to process')
    parser.add_argument('final_snap_nr', type=int, help='Index of the final snapshot')
    parser.add_argument('output_file', help='Where to write the output')
    args = parser.parse_args()
    
    message("Reading VR catalogue")
    mf = phdf5.MultiFile(args.vr_format, file_nr_dataset="Num_of_files", comm=comm)
    data = mf.read(("ID", "ID_mbp_bh"))
    subhalo_id = data["ID"]
    subhalo_id_mbp_bh = data["ID_mbp_bh"]

    message("Finding tracers")
    membership_cache = {}
    tracer_id = choose_bh_tracer(subhalo_id, args.snap_nr, args.final_snap_nr, args.snapshot_format,
                                 args.membership_format, membership_cache)

    # Count how many halos changed their most bound BH ID due to this process
    nr_groups = comm.allreduce(len(subhalo_id))
    nr_groups_changed = comm.allreduce(np.sum(tracer_id != subhalo_id_mbp_bh))
    fraction_changed = nr_groups_changed / nr_groups
    message(f"Fraction of groups which changed tracer BH ID = {fraction_changed}")

    #message("Writing results")
    #with h5py.File(args.outfile, "w", driver="mpio", comm=comm) as f:
    #    phdf5.collective_write(f, "tracer_id", tracer_id, comm)


def run():

    parser = MPIArgumentParser(comm, description='Create lightcone halo catalogues.')
    parser.add_argument('tree_filename',  help='Location of merger tree file')
    parser.add_argument('lightcone_dir',  help='Directory with lightcone particle outputs')
    parser.add_argument('lightcone_base', help='Base name of the lightcone to use')
    parser.add_argument('snapshot_format',  help='Format string for snapshot filenames (e.g. "snap_{snap_nr:04d}.{file_nr}.hdf5")')
    parser.add_argument('membership_format', help='Format string for group membership filenames')
    parser.add_argument('output_dir',     help='Where to write the output')
    args = parser.parse_args()
    match_black_holes(args)


if __name__ == "__main__":
    
    #run()
    test_choose_bh_tracers()
