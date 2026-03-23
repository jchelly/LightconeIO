#!/bin/env python

import numpy as np
import h5py
import os


def particle_file_name(basedir, basename, rank_nr):
    return f"{basedir}/{basename}_particles/{basename}_0000.{rank_nr}.hdf5"


def create_virtual_file(basedir, basename, outname):
    """
    Create a virtual file which can be used to access the lightcone
    particles. Must be run on post-processed (indexed) particle
    outputs.
    """

    # Create the output file
    outfile = h5py.File(outname, "w")
    print(f"Created output file {outname}")

    # Open one of the input files
    inname = particle_file_name(basedir, basename, rank_nr=0)
    infile0 = h5py.File(inname, "r")
    print(f"Opened input file {inname}")

    # Copy metadata
    for name in infile0:
        if name in ("Cells", "InternalCodeUnits", "Units", "Lightcone"):
            # These groups should be copied over
            print(f"  Copying {name}")
            infile0.copy(name, outfile)
        else:
            # Others are just created empty
            print(f"  Creating {name}")
            outfile.create_group(name)

    # Fix up attributes
    for name in outfile["Lightcone"].attrs:
        if name.startswith("cumulative"):
            del outfile["Lightcone"].attrs[name]
    outfile["Lightcone"].attrs["mpi_rank"] = (0,)
    outfile["Lightcone"].attrs["nr_mpi_ranks"] = (1,)
    outfile["Lightcone"].attrs["file_index"] = (0,)

    # Determine what particle types we have
    all_types = ("Gas", "DM", "Stars", "BH", "Neutrino")
    part_types = [pt for pt in all_types if pt in infile0]

    # Fix up the number of particles per file for each type
    num_per_file = {}
    for part_type in part_types:

        # Read particle counts
        first_particle_in_file = infile0[f"Cells/{part_type}/first_particle_in_file"][...]
        num_particles_in_file  = infile0[f"Cells/{part_type}/num_particles_in_file"][...]
        total_nr = np.sum(num_particles_in_file, dtype=num_particles_in_file.dtype)
        num_per_file[part_type] = num_particles_in_file
        print(f"There are {total_nr} particles of type {part_type}")

        # Compute new values assuming all particles are in a single file
        dtype = num_particles_in_file.dtype
        first_particle_in_file = np.zeros(1, dtype=dtype)
        num_particles_in_file  = np.ones(1, dtype=dtype) * total_nr

        # Write new values
        del outfile[f"Cells/{part_type}/first_particle_in_file"]
        outfile[f"Cells/{part_type}/first_particle_in_file"] = first_particle_in_file
        del outfile[f"Cells/{part_type}/num_particles_in_file"]
        outfile[f"Cells/{part_type}/num_particles_in_file"] = num_particles_in_file

    # Determine relative path from the output file's directory to the input
    relative_path = os.path.relpath(inname, start=os.path.dirname(outname))
    relative_dir = os.path.dirname(relative_path)
    print(f"Relative path to input = {relative_dir}")

    # Loop over particle types
    for part_type in part_types:
        print(f"Create datasets for type {part_type}")

        # Loop over datasets for this particle type
        for dset_name in infile0[part_type]:

            # Get data type and dimensions of this dataset
            dset_in = infile0[part_type][dset_name]
            shape_in = dset_in.shape
            dtype = dset_in.dtype

            # Compute full shape of the result
            shape_out = list(shape_in)
            shape_out[0] = int(sum(num_per_file[part_type]))
            shape_out = tuple(shape_out)
            print(f"  {dset_name} has dtype={dtype}, shape={shape_out}")
            # Create the virtual layout
            layout = h5py.VirtualLayout(shape=shape_out, dtype=dtype)

            # Add all of the sub files to the layout
            nr_files = len(num_per_file[part_type])
            offset = 0
            for file_nr in range(nr_files):

                # Get the name of the next file
                filename = f"{relative_dir}/{basename}_0000.{file_nr}.hdf5"

                # Find the dimensions of the dataset in the file
                shape_this_file = list(shape_in)
                shape_this_file[0] = int(num_per_file[part_type][file_nr])
                shape_this_file = tuple(shape_this_file)

                # Update the virtual layout
                count = num_per_file[part_type][file_nr]
                layout[offset:offset+count,...] = h5py.VirtualSource(filename, f"{part_type}/{dset_name}", shape=shape_this_file)
                offset += count

            # Create the virtual dataset
            vdset = outfile.create_virtual_dataset(f"{part_type}/{dset_name}", layout, fillvalue=-999)

            # Copy any attributes
            for attr_name, attr_val in infile0[f"{part_type}/{dset_name}"].attrs.items():
                vdset.attrs[attr_name] = attr_val


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Create virtual file to combine SWIFT particle lightcone outputs.")
    parser.add_argument("basedir", type=str, help="Lightcone output directory containing index files")
    parser.add_argument("basename", type=str, help="Name of the lightcone to process (e.g. lightcone0)")
    parser.add_argument("outname", type=str, help="Name of the virtual file to create")
    args = parser.parse_args()

    create_virtual_file(args.basedir, args.basename, args.outname)
