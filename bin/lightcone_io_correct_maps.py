#!/bin/env python

import subprocess
import sys
import os
import shutil
import numpy as np
import h5py
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def correct_shell_file(indir, basename, outdir, shell_nr, z_central):

    # Generate shell input and output file names
    infile = ("%s/%s_shells/shell_%d/%s.shell_%d.0.hdf5" %
              (indir, basename, shell_nr, basename, shell_nr))
    outfile = ("%s/%s_shells/shell_%d/%s.shell_%d.0.hdf5" %
               (outdir, basename, shell_nr, basename, shell_nr))

    # Create the output directory
    try:
        os.makedirs(os.path.dirname(outfile))
    except OSError:
        pass

    # Copy the file
    #shutil.copyfile(infile, outfile)
    subprocess.run(["cp", infile, outfile], check=True)

    #
    # Now need to apply corrections
    #
    f = h5py.File(outfile, "r+")

    # DopplerB needs to be multiplied by (1+z)
    if "DopplerB" in f:
        dset = f["DopplerB"]
        if "Correction factor" in dset.attrs:
            raise Exception("Dataset has already been corrected!")
        fac = (1.0+z_central[shell_nr])
        dset.attrs["Correction applied"] = "Multiplied by 1+(z_min+z_max)/2 to compensate for bug in original output"
        dset.attrs["Correction factor"] = fac
        dset.attrs["Central redshift assumed for correction"] = z_central[shell_nr]
        data = dset[...]
        data *= fac
        dset[...] = data

    # DM needs to be multiplied by 1/(1+z)
    if "DM" in f:
        dset = f["DM"]
        if "Correction factor" in dset.attrs:
            raise Exception("Dataset has already been corrected!")
        fac = 1.0/(1.0+z_central[shell_nr])
        dset.attrs["Correction applied"] = "Multiplied by 1/(1+(z_min+z_max)/2) to compensate for bug in original output"
        dset.attrs["Correction factor"] = fac
        dset.attrs["Central redshift assumed for correction"] = z_central[shell_nr]
        data = dset[...]
        data *= fac
        dset[...] = data

    f.close()
    print("Shell %d done" % shell_nr)


def correct_maps_mpi(indir, shell_redshifts, outdir, basenames):

    if comm_rank == 0:

        # Read in the shell redshifts
        with open(shell_redshifts, "r") as f:
            header = f.readline()
            # Check for the expected header
            if header.strip() != "# Minimum redshift, Maximum redshift":
                raise Exception("Shell radii must be specified as redshifts!")
            # Read radii
            z_central = []
            while True:
                line = f.readline().strip()
                if len(line) == 0:
                    break
                else:
                    z_central.append(0.5*sum([float(col) for col in line.split(",")]))

        z_central = np.asarray(comm.bcast(z_central), dtype=float)
        print("Found %d shell redshifts" % len(z_central))
    
        # Accumulate correct_shell_file() arguments needed to process each shell
        args = []
        for basename in basenames:

            # Find number of shells in this lightcone
            with h5py.File("%s/%s_index.hdf5" % (indir, basename), "r") as infile:
                nr_shells = int(infile["Lightcone"].attrs["nr_shells"])
                nr_files_per_shell = int(infile["Lightcone"].attrs["nr_files_per_shell"])
            if nr_files_per_shell != 1:
                raise Exception("Can only correct single file outputs!")

            # Ensure the output directory exists
            try:
                os.makedirs(outdir)
            except OSError:
                pass

            # Copy the index file to the output
            infile =  "%s/%s_index.hdf5" % (indir, basename)
            outfile = "%s/%s_index.hdf5" % (outdir, basename)
            shutil.copyfile(infile, outfile)

            # Store arguments to process this shell
            args += [(indir, basename, outdir, shell_nr, z_central) for shell_nr in range(nr_shells)]

    # Make sure all output dirs have been created
    comm.barrier()

    # Apply corrections to all shells in parallel
    with MPICommExecutor(comm, root=0) as executor:
        if executor is not None:
            executor.starmap(correct_shell_file, args)

    comm.barrier()
    print("All lightcones done.")


if __name__ == "__main__":
    
    usage="""
    Usage: python3 -m mpi4py lightcone_io_correct_maps.py \\
              basedir shell_redshifts outdir basename1 basename2 ...

    basedir      : location of the input lightcones
    shell_redshifts : name of file with the shell redshifts
    outdir       : where to write the output
    basename*    : basenames of lightcones to process
"""

    args = {}
    if comm_rank == 0:
        if len(sys.argv) < 4:
            print(usage)
            args = None
        else:
            args["indir"] = sys.argv[1]
            args["shell_redshifts"] = sys.argv[2]
            args["outdir"] = sys.argv[3]
            args["basenames"] = sys.argv[4:]
    args = comm.bcast(args)

    if args is None:
        MPI.Finalize()
        sys.exit(0)

    correct_maps_mpi(args["indir"], args["shell_redshifts"], args["outdir"], args["basenames"])

    comm.barrier()
    if comm_rank == 0:
        print("All shells done for lightcones ", args["basenames"])
