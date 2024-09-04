#!/bin/env python

import os
import sys
import numpy as np
import h5py
import argparse
from mpi4py import MPI
import unyt

import virgo.util.match as match
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from virgo.mpi.util import MPIArgumentParser
import virgo.formats.swift

import lightcone_io.particle_reader as pr
import lightcone_io.halo_catalogue as hc
import lightcone_io.choose_bh_tracer as ct

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Special most bound black hole ID for halos with no black holes.
# Must not appear as a particle ID.
NULL_BH_ID = np.iinfo(np.int64).max

# Particle type names used in the lightcone
part_type_names = [
    "Gas",
    "DM",
    None,
    None,
    "Stars",
    "BH",
    "Neutrino",
    ]


def attributes_from_units(units):
    """
    Given a unyt.Unit object, generate SWIFT dataset attributes

    units: the Unit object

    Returns a dict with the attributes
    """
    attrs = {}

    # Get CGS conversion factor. Note that this is the conversion to physical units,
    # because unyt multiplies out the dimensionless a factor.
    cgs_factor, offset = units.get_conversion_factor(units.get_cgs_equivalent())

    # Get a exponent
    a_unit = unyt.Unit("a", registry=units.registry)
    a_exponent = units.expr.as_powers_dict()[a_unit.expr]
    a_val = a_unit.base_value

    # Get h exponent
    h_unit = unyt.Unit("h", registry=units.registry)
    h_exponent = units.expr.as_powers_dict()[h_unit.expr]
    h_val = h_unit.base_value

    # Find the power associated with each dimension
    powers = units.get_mks_equivalent().dimensions.as_powers_dict()

    # Set the attributes
    attrs["Conversion factor to CGS (not including cosmological corrections)"] = [
        float(cgs_factor / (a_val ** a_exponent) / (h_val ** h_exponent))
    ]
    attrs["Conversion factor to physical CGS (including cosmological corrections)"] = [
        float(cgs_factor)
    ]
    attrs["U_I exponent"] = [float(powers[unyt.dimensions.current_mks])]
    attrs["U_L exponent"] = [float(powers[unyt.dimensions.length])]
    attrs["U_M exponent"] = [float(powers[unyt.dimensions.mass])]
    attrs["U_T exponent"] = [float(powers[unyt.dimensions.temperature])]
    attrs["U_t exponent"] = [float(powers[unyt.dimensions.time])]
    attrs["a-scale exponent"] = [float(a_exponent)]
    attrs["h-scale exponent"] = [float(h_exponent)]

    return attrs


def message(m):
    if comm_rank == 0:
        print(m)


def drop_a_from_comoving_length(arr):
    """
    Convert an array of lengths to comoving snapshot length units and drop
    the a factor from the units without changing the value.

    Coordinates in the lightcone are comoving but the expansion factor varies
    so we can't include it explicitly in the units as we might in snapshots.
    Here we convert to units a*snap_length and reinterpret the numbers as
    just snap_length.
    """
    reg = arr.units.registry
    return arr.to("a*snap_length").value * unyt.Unit("snap_length", registry=reg)
    

def match_black_holes(args):

    message(f"Starting on {comm_size} MPI ranks")

    # Determine quantities to read from the halo catalogue
    to_read = ["InputHalos/HaloCentre", "InputHalos/HaloCatalogueIndex"]
    if args.pass_through is not None:
        for prop_name in args.pass_through.split(","):
            to_read.append(prop_name)
    
    # Open the lightcone particle output in MPI mode
    filename = f"{args.lightcone_dir}/{args.lightcone_base}_particles/{args.lightcone_base}_0000.0.hdf5"
    lightcone = pr.IndexedLightcone(filename, comm=comm)

    # Get simulation box size and unit system from a snapshot file.
    # HBT output specifies how to convert its chosen units to Mpc and Msolar so
    # we need the snapshot unit system to interpret the catalogues using SWIFT's
    # physical constants for consistency.
    if comm_rank == 0:
        filename = args.snapshot_format.format(snap_nr=args.last_sim_snap, file_nr=0)
        with h5py.File(filename, "r") as infile:
            boxsize_no_units = infile["Header"].attrs["BoxSize"][0]
            swift_unit_registry = virgo.formats.swift.soap_unit_registry_from_snapshot(infile)
    else:
        boxsize_no_units = None
        swift_unit_registry = None
    boxsize_no_units, swift_unit_registry = comm.bcast((boxsize_no_units, swift_unit_registry))

    # Assign units to the boxsize. Box size is comoving but we deliberately
    # omit the a factor here.
    boxsize = boxsize_no_units * unyt.Unit("snap_length", registry=swift_unit_registry)
    
    # Open the halo catalogue
    if args.halo_type == "SOAP":
        halo_cat = hc.SOAPCatalogue(args.halo_format, args.first_sim_snap, args.last_sim_snap)
    elif args.halo_type == "HBTplus":
        halo_cat = hc.HBTplusCatalogue(args.halo_format, args.snapshot_format, args.first_sim_snap, args.last_sim_snap)
    else:
        raise ValueError("Unrecognized value for --halo-type option")
                    
    # Loop over snapshots
    halos_so_far = 0
    membership_cache = {}
    for snap_nr in range(args.end_snap, args.start_snap-1, -1):

        # Read halos at this snapshot
        halo_data = halo_cat.read(snap_nr, to_read)
        
        # Count halos
        nr_halos_in_slice = len(halo_data["InputHalos/HaloCatalogueIndex"])
        nr_halos_in_slice_all = comm.allreduce(nr_halos_in_slice)
        
        # Choose the tracer BH particle to use for each object.
        # Returns ID and position of the selected BH particle.
        part_type = f"PartType{args.part_type}"
        message(f"  Choosing tracer particles for snapshot {snap_nr} using {part_type}")
        tracer_id, tracer_pos = ct.choose_bh_tracer(halo_data["InputHalos/HaloCatalogueIndex"],
                                                    snap_nr, args.last_sim_snap, args.snapshot_format,
                                                    args.membership_format, membership_cache,
                                                    part_type, NULL_BH_ID)
        tracer_pos = drop_a_from_comoving_length(tracer_pos)
        
        # Each snapshot populates a redshift range which reaches half way to adjacent snapshots
        # (range is truncated for the first and last snapshots)
        z_snap = halo_cat.redshift[snap_nr]        
        if snap_nr == args.first_sim_snap:
            z2 = halo_cat.redshift[args.first_sim_snap]
        else:
            z2 = 0.5*(halo_cat.redshift[snap_nr-1]+halo_cat.redshift[snap_nr])
        if snap_nr == args.last_sim_snap:
            z1 = halo_cat.redshift[args.last_sim_snap]
        else:
            z1 = 0.5*(halo_cat.redshift[snap_nr]+halo_cat.redshift[snap_nr+1])

        message(f"  Using {nr_halos_in_slice_all} halos at z={z_snap:.3f} to populate range z={z1:.3f} to z={z2:.3f}")

        # Read in the lightcone BH particle positions and IDs in this redshift range
        lightcone_props = ("Coordinates", "ParticleIDs", "ExpansionFactors")
        part_type_name = part_type_names[args.part_type]
        particle_data = lightcone[part_type_name].read_exact(lightcone_props, redshift_range=(z1,z2))
        if np.any(particle_data["ParticleIDs"] == NULL_BH_ID):
            raise RuntimeError("Found a BH particle with ID=NULL_BH_ID!")
        nr_parts = len(particle_data["ParticleIDs"])
        nr_parts_all = comm.allreduce(nr_parts)
        message(f"  Read in {nr_parts_all} lightcone BH particles for this redshift range")

        # Try to match BH particles to the halo most bound BH IDs.
        # There may be multiple particles matching each halo due to the periodicity of the box.
        # Since halos with no black hole have tracer_id=NULL_BH_ID and this value never appears
        # in the particle data, every match will become a halo in the output catalogue.
        halo_index = psort.parallel_match(particle_data["ParticleIDs"], tracer_id, comm=comm)
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
        for name in sorted(halo_data):
            halo_slice[name] = psort.fetch_elements(halo_data[name], halo_index, comm=comm)
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
        registry = tracer_pos.units.registry
        length_unit = unyt.Unit("snap_length", registry=registry)
        
        # Position of the matched BH particle in the lightcone particle output.
        # Has one entry for each BH in the lightcone which matched a halo.
        bh_pos_in_lightcone = particle_data["Coordinates"][matched,...].to(length_unit)

        # Position of the selected tracer BH, taken from the snapshot.
        bh_pos_in_snapshot = psort.fetch_elements(tracer_pos, halo_index, comm=comm).to(length_unit)
                
        # Position of the matched halo from the halo finder:
        # Note that the units include an a factor, which we need to remove
        halo_pos_in_snapshot = drop_a_from_comoving_length(halo_slice["InputHalos/HaloCentre"]).to(length_unit)
        
        # Vector from the tracer BH to the potential minimum - may need box wrapping
        bh_to_minpot_vector = halo_pos_in_snapshot - bh_pos_in_snapshot
        bh_to_minpot_vector = ((bh_to_minpot_vector+0.5*boxsize) % boxsize) - 0.5*boxsize

        # Compute halo potential minimum position in lightcone
        halo_pos_in_lightcone = bh_pos_in_lightcone + bh_to_minpot_vector

        # Report largest offset between BH and potential minimum
        max_offset = ct.distributed_amax(bh_to_minpot_vector.flatten(), comm)
        if comm_rank == 0 and max_offset is not None:
            message(f"  Maximum minpot/bh offset = {max_offset} (SWIFT length units, comoving)")
            
        # Add the position and redshift in the lightcone to the output catalogue
        halo_slice["Lightcone/HaloCentre"] = halo_pos_in_lightcone
        #halo_slice["Lightcone/TracerPosition"] = bh_pos_in_lightcone
        halo_slice["Lightcone/Redshift"] = unyt.unyt_array(1.0/particle_data["ExpansionFactors"][matched]-1.0,
                                                           units="dimensionless", registry=registry)
        # Add snapshot number to the output catalogue
        snap_nr_arr = np.ones(len(halo_slice["Lightcone/Redshift"]), dtype=int)*snap_nr
        halo_slice["Lightcone/SnapshotNumber"] = unyt.unyt_array(snap_nr_arr, dtype=int, units="dimensionless", registry=registry)
        
        message(f"  Computed potential minimum position in lightcone")
        
        # Write out the halo catalogue for this snapshot
        output_filename = f"{args.output_dir}/lightcone_halos_{snap_nr:04d}.hdf5"
        outfile = h5py.File(output_filename, "w", driver="mpio", comm=comm, libver="v108")
        for name in sorted(halo_slice):
            # Ensure the group exists
            outfile.require_group(os.path.dirname(name))
            # Write the data
            dset = phdf5.collective_write(outfile, name, halo_slice[name], gzip=6, comm=comm)
            # Write units
            attrs = attributes_from_units(halo_slice[name].units)
            for attr_name, attr_val in attrs.items():
                dset.attrs[attr_name] = attr_val
            # Write description
            dset.attrs["Description"] = halo_cat.description[name]
                
        # Correct a-exponent of the lightcone positions (they're comoving)
        outfile["Lightcone/HaloCentre"].attrs["a-scale exponent"] = (1.0,)
        outfile.close()
        message(f"  Wrote file: {output_filename}")

        # Tidy up particle arrays before we read the next slice
        del particle_data

    comm.barrier()
    message("All redshift ranges done.")


def run():

    parser = MPIArgumentParser(comm, description='Create lightcone halo catalogues.')
    parser.add_argument('halo_format', help='Format string for halo catalogue filenames (using {snap_nr}, {file_nr})')
    parser.add_argument('first_sim_snap', type=int, help='Index of the first snapshot of the simulation')
    parser.add_argument('last_sim_snap', type=int, help='Index of the last snapshot of the simulation')
    parser.add_argument('start_snap', type=int, help='Index of the first snapshot to process')
    parser.add_argument('end_snap', type=int, help='Index of the last snapshot to process')
    parser.add_argument('lightcone_dir',  help='Directory with lightcone particle outputs')
    parser.add_argument('lightcone_base', help='Base name of the lightcone to use')
    parser.add_argument('snapshot_format',  help='Format string for snapshot filenames (e.g. "snap_{snap_nr:04d}.{file_nr}.hdf5")')
    parser.add_argument('membership_format', help='Format string for group membership filenames')
    parser.add_argument('output_dir',     help='Where to write the output')
    parser.add_argument('--pass-through', default=None, help='Comma separated list of SOAP properties to pass through')
    parser.add_argument('--halo-type', choices=("SOAP","HBTplus"), default="SOAP", help='Input halo catalogue type')
    parser.add_argument('--part-type', type=int, default=5, help='Particle type to use for placing lightcone halos')
    args = parser.parse_args()
    match_black_holes(args)


if __name__ == "__main__":
    run()
