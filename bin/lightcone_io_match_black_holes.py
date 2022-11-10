#!/bin/env python

import os
import sys
import numpy as np
import h5py
import argparse
from mpi4py import MPI

import virgo.util.match as match
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
import lightcone_io.particle_reader as pr

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


class ArgumentParserError(Exception): pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(message+"\n")
        raise ArgumentParserError(message)


def message(m):
    if comm_rank == 0:
        print(m)


def match_black_holes(args):

    if comm_rank > 0:
        sys.stderr = open('/dev/null', 'w')

    message(f"Starting on {comm_size} MPI ranks")

    # Open the lightcone particle output in MPI mode
    filename = f"{args.lightcone_dir}/{args.lightcone_base}_particles/{args.lightcone_base}_0000.0.hdf5"
    lightcone = pr.IndexedLightcone(filename, comm=comm)

    # Get snapshot redshifts from the tree file
    if comm_rank == 0:
        with h5py.File(args.tree_filename % {"file_nr" : 0}, "r") as treefile:
            output_snapshots = treefile["Snapshots/SnapNum"][...]
            output_redshifts = treefile["Snapshots/Redshift"][...]
        max_snap_nr = np.amax(output_snapshots)
        redshifts = -np.ones(max_snap_nr+1, dtype=float)
        for outs, outr in zip(output_snapshots, output_redshifts):
            redshifts[outs] = outr
    else:
        redshifts = None
    redshifts = comm.bcast(redshifts)

    # Read the merger tree data we need:
    # These will all be passed through to the output.
    # The Redshift and [XYZ]cminpot arrays will be updated to the point of lightcone crossing.
    merger_tree_props = ("Subhalo/Xcmbp_bh", "Subhalo/Ycmbp_bh", "Subhalo/Zcmbp_bh",
                         "Subhalo/Xcminpot", "Subhalo/Ycminpot", "Subhalo/Zcminpot",
                         "Subhalo/ID_mbp_bh", "Subhalo/n_bh", "Subhalo/Structuretype", 
                         "Subhalo/SnapNum", "Subhalo/ID",
                         "Subhalo/Mass_tot", "Subhalo/Mass_star",
                         "Subhalo/Mass_gas", "Subhalo/Mass_bh")
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
    for redshift_nr in range(len(redshifts)):
        z = merger_tree["Subhalo/Redshift"][first_at_redshift[redshift_nr]:first_at_next_redshift[redshift_nr]]
        if not(np.all(z==redshifts[redshift_nr])):
            raise RuntimeError("Redshift ranges not identified correctly!")
    message("Identified range of halos at each redshift")

    # Special ID used to indicate no black hole. Will check that this doesn't appear as a real ID.
    NULL_BH_ID = 0
    no_bh = merger_tree["Subhalo/n_bh"]==0
    assert np.all(merger_tree["Subhalo/ID_mbp_bh"][no_bh] == NULL_BH_ID)

    # Loop over unique redshifts in the trees, excluding the last
    halos_so_far = 0
    for redshift_nr in range(len(redshifts)-1):
        
        # Find redshift range and range of halos for this iteration
        z1 = redshifts[redshift_nr]
        z2 = redshifts[redshift_nr+1]
        i1 = first_at_redshift[redshift_nr]
        i2 = first_at_next_redshift[redshift_nr]
        nr_halos_in_slice = i2-i1
        nr_halos_in_slice_all = comm.allreduce(nr_halos_in_slice)
        message(f"Processing {nr_halos_in_slice_all} halos in redshift range {z1:.2f} to {z2:.2f}")

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

        # Compute the position of each halo in the output:
        #
        # We know:
        #  - the position in the lightcone of the most bound black hole particle
        #  - the position in the snapshot of the most bound black hole particle
        #  - the position in the snapshot of the halo's potential minimum
        #
        # We want to compute the position in the lightcone of the potential minimum.
        #
        bh_pos_in_lightcone = particle_data["Coordinates"][matched,...].ndarray_view()
        bh_pos_in_snapshot  = np.column_stack((halo_slice["Subhalo/Xcmbp_bh"],
                                               halo_slice["Subhalo/Ycmbp_bh"],
                                               halo_slice["Subhalo/Zcmbp_bh"]))
        halo_pos_in_snapshot = np.column_stack((halo_slice["Subhalo/Xcminpot"],
                                                halo_slice["Subhalo/Ycminpot"],
                                                halo_slice["Subhalo/Zcminpot"]))
        halo_pos_in_lightcone = bh_pos_in_lightcone + (halo_pos_in_snapshot - bh_pos_in_snapshot)

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

        # Write out the halo catalogue for this snapshot
        output_filename = f"{args.output_dir}/lightcone_halos_{redshift_nr:04d}.hdf5"
        outfile = h5py.File(output_filename, "w", driver="mpio", comm=comm)
        outfile.create_group("Subhalo")
        for name in halo_slice:
            writebuf = np.ascontiguousarray(halo_slice[name])
            phdf5.collective_write(outfile, name, writebuf, comm=comm)
        outfile.close()

        # Count halos output so far, including this redshift slice
        halos_so_far_all = comm.allreduce(halos_so_far)

        # Add the completeness information etc
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

    message("All redshift ranges done.")


if __name__ == "__main__":

    # Get command line arguments
    if comm.Get_rank() == 0:
        os.environ['COLUMNS'] = '80' # Can't detect terminal width when running under MPI?
        parser = ThrowingArgumentParser(description='Create lightcone halo catalogues.')
        parser.add_argument('tree_filename',  help='Location of merger tree file')
        parser.add_argument('lightcone_dir',  help='Directory with lightcone particle outputs')
        parser.add_argument('lightcone_base', help='Base name of the lightcone to use')
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

    match_black_holes(args)

