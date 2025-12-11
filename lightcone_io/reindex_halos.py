#!/bin/env python
#
# Reorder a halo lightcone output and add spatial indexing.
# Makes a sorted copy of the input.
#

import os
import os.path
import numpy as np
import h5py
import healpy as hp

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort


# HDF5 chunk size for the output
chunk_size = 1024*1024


def message(m):
    if comm_rank == 0:
        print(m)


def find_groups(group):
    all_groups = []
    def store_name(name, obj):
        if isinstance(obj, h5py.Group):
            all_groups.append(name)
    group.visititems(store_name)
    return all_groups


def find_datasets(group):
    all_datasets = []
    def store_name(name, obj):
        if isinstance(obj, h5py.Dataset) and (not name.startswith("Index/")):
            all_datasets.append(name)
    group.visititems(store_name)
    return all_datasets


def reindex_halos(snap_nr, input_lightcone_dir, lightcone_base,
                  output_lightcone_dir, nside, order, soap_format):

    # Get paths to the input and output files
    input_filename = f"{input_lightcone_dir}/{lightcone_base}/lightcone_halos_{snap_nr:04d}.hdf5"
    output_filename = f"{output_lightcone_dir}/{lightcone_base}/lightcone_halos_{snap_nr:04d}.hdf5"

    # Ensure the output directory exists
    if comm_rank == 0:
        outdir = os.path.dirname(output_filename)
        os.makedirs(outdir, exist_ok=True)
    comm.barrier()

    message(f"Creating output file for snapshot {snap_nr}")
    if comm_rank == 0:
        with h5py.File(output_filename, "w") as outfile, h5py.File(input_filename, "r") as infile:
            # Store names of groups in the input
            group_names = find_groups(infile)
            # Create groups in the output and copy any attributes
            for name in group_names:
                group = outfile.require_group(name)
                for attr_name, attr_val in infile[name].attrs.items():
                    group.attrs[attr_name] = attr_val
            # Overwrite nside and order with new values, if present
            outfile.require_group("Index")
            outfile["Index"].attrs["nside"] = nside
            outfile["Index"].attrs["order"] = order
            # Get list of dataset names from the input file
            dataset_names = find_datasets(infile)
    else:
        group_names = None
        dataset_names = None
    group_names, dataset_names = comm.bcast((group_names, dataset_names))

    message(f"Reading halo positions")
    with h5py.File(input_filename, "r", driver="mpio", comm=comm) as infile:
        pos = phdf5.collective_read(infile["Lightcone/HaloCentre"], comm)

    message(f"Computing pixel index for each halo")
    pixel_index = hp.pixelfunc.vec2pix(nside, pos[:,0], pos[:,1], pos[:,2], nest=(order=="nest"))

    message("Computing halo ordering by pixel index")
    order = psort.parallel_sort(pixel_index, return_index=True, comm=comm)

    message("Counting number of halos in each pixel")
    npix = hp.pixelfunc.nside2npix(nside)
    halos_per_pixel = psort.parallel_bincount(pixel_index, minlength=npix, comm=comm)
    del pixel_index

    message("Computing offset to first halo in each pixel")
    total_halos_this_rank = np.sum(halos_per_pixel, dtype=np.int64)
    total_halos_prev_ranks = comm.scan(total_halos_this_rank) - total_halos_this_rank
    first_halo_in_pixel = np.cumsum(halos_per_pixel, dtype=np.int64) - halos_per_pixel + total_halos_prev_ranks

    # Write sorted datasets to the output file
    with (h5py.File(input_filename, "r", driver="mpio", comm=comm) as infile,
          h5py.File(output_filename, "r+", driver="mpio", comm=comm) as outfile):
        message("Writing index information")
        phdf5.collective_write(outfile["Index"], "FirstHaloInPixel", first_halo_in_pixel, gzip=6, chunk=chunk_size, comm=comm)
        phdf5.collective_write(outfile["Index"], "NumHalosPerPixel", halos_per_pixel, gzip=6, chunk=chunk_size, comm=comm)
        for dataset_name in dataset_names:
            message(f"  Reordering dataset: {dataset_name}")
            data = phdf5.collective_read(infile[dataset_name], comm)
            data = psort.fetch_elements(data, order, comm=comm)
            phdf5.collective_write(outfile, dataset_name, data, gzip=6, chunk=chunk_size, comm=comm)

    # Add SOAP index, if not already present and we have a SOAP output
    property_names = dataset_names.copy()
    if soap_format is not None and "InputHalos/SOAPIndex" not in dataset_names:
        comm.barrier()
        message("Reading TrackId from SOAP catalogue")
        soap_filename = soap_format.format(snap_nr=snap_nr)
        with h5py.File(soap_filename, "r", driver="mpio", comm=comm) as soap_file:
            soap_trackid = phdf5.collective_read(soap_file["InputHalos/HBTplus/TrackId"], comm)
        message("Reading TrackId from halo lightcone")
        with h5py.File(input_filename, "r", driver="mpio", comm=comm) as soap_file:
            lightcone_trackid = phdf5.collective_read(soap_file["InputHalos/HBTplus/TrackId"], comm)
        message("Computing index of each lightcone halo in SOAP")
        soap_index = psort.parallel_match(lightcone_trackid, soap_trackid, comm=comm)
        assert np.all(soap_index >= 0)
        message("Writing soap index to halo lightcone")
        with h5py.File(output_filename, "r+", driver="mpio", comm=comm) as outfile:
            dataset = phdf5.collective_write(outfile, "InputHalos/SOAPIndex", soap_index, gzip=6, chunk=chunk_size, comm=comm)
            # The input doesn't include this dataset, so can't copy attributes
            dataset.attrs["Description"] = "Index of the halo in the input SOAP catalogue"
            dataset.attrs["Conversion factor to CGS (not including cosmological corrections)"] = [1.0]
            dataset.attrs["Conversion factor to CGS (including cosmological corrections)"] = [1.0]
            for dim in ("U_I", "U_L", "U_M", "U_T", "U_t", "a-scale", "h-scale"):
                dataset.attrs[f"{dim} exponent"] = [0.0]
        property_names.append("InputHalos/SOAPIndex")

    comm.barrier()
    message("Copying dataset attributes")
    if comm_rank == 0:
        with (h5py.File(input_filename, "r") as infile,
              h5py.File(output_filename, "r+") as outfile):
            for dataset_name in dataset_names:
                input_dataset = infile[dataset_name]
                output_dataset = outfile[dataset_name]
                for attr_name, attr_val in input_dataset.attrs.items():
                    output_dataset.attrs[attr_name] = attr_val
            outfile["Lightcone"].attrs["property_names"] = property_names

    message(f"Snapshot {snap_nr} done.")


def run():

    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(comm, description='Reorder lightcone halo catalogues.')
    parser.add_argument('start_snap', type=int, help='Index of the first snapshot to process')
    parser.add_argument('end_snap', type=int, help='Index of the last snapshot to process')
    parser.add_argument('input_lightcone_dir',  help='Directory with lightcone particle outputs')
    parser.add_argument('lightcone_base', help='Base name of the lightcone to use')
    parser.add_argument('output_lightcone_dir',  help='Directory with lightcone particle outputs')
    parser.add_argument("--nside", type=int, default=16, help="HEALPpix map resolution to use to bin halos in the output")
    parser.add_argument("--order", choices=["nest","ring"], default="nest", help="HEALPix pixel ordering scheme")
    parser.add_argument("--soap-format", type=str, default=None, help="Format string to generate SOAP filenames (use {snap_nr})")
    args = parser.parse_args()

    for snap_nr in range(args.start_snap, args.end_snap+1):
        reindex_halos(snap_nr, args.input_lightcone_dir, args.lightcone_base,
                      args.output_lightcone_dir, args.nside, args.order,
                      args.soap_format)

if __name__ == "__main__":
    run()
