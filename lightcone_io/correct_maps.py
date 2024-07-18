#!/bin/env python

import subprocess
import sys
import os
import shutil
import numpy as np
import h5py
import yaml
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def correct_shell_file(indir, basename, outdir, shell_nr, z_central, neutrino_density):

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

    # Check that the input file exists
    if not(os.path.exists(infile)):
        raise RuntimeError(f"Input file {infile} does not exist!")

    # Copy the file
    #shutil.copyfile(infile, outfile)
    #subprocess.run(["cp", infile, outfile], check=True)

    # Move the file
    subprocess.run(["mv", infile, outfile], check=True)

    #
    # Now need to apply corrections
    #
    f = h5py.File(outfile, "r+")

    # DopplerB needs to be multiplied by (1+z)
    if "DopplerB" in f:
        dset = f["DopplerB"]
        if "Correction factor" not in dset.attrs:
            fac = (1.0+z_central[shell_nr])
            dset.attrs["Correction applied"] = "Multiplied by 1+(z_min+z_max)/2 to compensate for bug in original output"
            dset.attrs["Correction factor"] = fac
            dset.attrs["Central redshift assumed for correction"] = z_central[shell_nr]
            nchunk=100*1024*1024
            ntot = dset.shape[0]
            for offset in range(0, ntot, nchunk):
                n = min(nchunk, ntot-offset)
                data = dset[offset:offset+n,...]
                data *= fac
                dset[offset:offset+n,...] = data
        else:
            print(f"  Skipping DopplerB correction for shell {shell_nr} of {basename}")

    # DM needs to be multiplied by 1/(1+z)
    if "DM" in f:
        dset = f["DM"]
        if "Correction factor" not in dset.attrs:
            fac = 1.0/(1.0+z_central[shell_nr])
            dset.attrs["Correction applied"] = "Multiplied by 1/(1+(z_min+z_max)/2) to compensate for bug in original output"
            dset.attrs["Correction factor"] = fac
            dset.attrs["Central redshift assumed for correction"] = z_central[shell_nr]
            nchunk=100*1024*1024
            ntot = dset.shape[0]
            for offset in range(0, ntot, nchunk):
                n = min(nchunk, ntot-offset)
                data = dset[offset:offset+n,...]
                data *= fac
                dset[offset:offset+n,...] = data
        else:
            print(f"  Skipping DM correction for shell {shell_nr} of {basename}")


    # Neutrino mass maps:
    # Need to subtract 3*(mean neutrino density) to get actual neutrino density.
    # Will need to correct the neutrino mass and total mass maps. The mean
    # neutrino density can be extracted from the snapshot file.
    for name in ("NeutrinoMass", "TotalMass"):
        if name in f:
            dset = f[name]
            if "Correction offset" not in dset.attrs:
                # Find the volume of a pixel
                r1 = dset.attrs["comoving_inner_radius"][0]
                r2 = dset.attrs["comoving_outer_radius"][0]
                npix = dset.attrs["number_of_pixels"]
                pixel_volume = (4./3.)*np.pi*(r2**3-r1**3)/npix
                # Compute offset to apply to the map
                mass_per_pixel = -3.0*(neutrino_density*pixel_volume)
                # Update attributes with correction to apply
                dset.attrs["Correction applied"] = ["Subtracted 3 times the mean neutrino mass per pixel to fix bug in original output"]
                dset.attrs["Correction offset"] = mass_per_pixel
                # Apply  the correction to the neutrino mass
                nchunk=100*1024*1024
                ntot = dset.shape[0]
                for offset in range(0, ntot, nchunk):
                    n = min(nchunk, ntot-offset)
                    data = dset[offset:offset+n,...]
                    data += mass_per_pixel
                    dset[offset:offset+n,...] = data
            else:
                print(f"  Skipping {name} correction for shell {shell_nr} of {basename}")

    f.close()
    print("Shell %d done" % shell_nr)


def correct_maps_mpi(indir, yml_file, outdir, snapfile, basenames):

    if comm_rank == 0:

        # Get simulation directory absolute path
        sim_dir = os.path.abspath(os.path.dirname(yml_file))

        # Read simulation config file
        config = yaml.load(open(yml_file, "r"), yaml.loader.SafeLoader)
        
        # Get mean comoving neutrino density from the snapshot
        with h5py.File(snapfile, "r") as snap:
            omega_nu_0 = snap["Cosmology"].attrs["Omega_nu_0"][0]
            H0 = snap["Cosmology"].attrs["H0 [internal units]"][0]
            G = snap["PhysicalConstants"]["InternalUnits"].attrs["newton_G"][0]
        critical_density_0 = 3*(H0**2)/(8*np.pi*G)
        neutrino_density = omega_nu_0 * critical_density_0

        # Find paths to the shell redshift lists:
        # May be a different file for each lightcone
        radius_file = {}
        z_central = {}
        for basename in basenames:
            radius_file[basename] = None
            for i in range(8):
                section = "Lightcone%d" % i
                if section in config and config[section]["basename"] == basename and "radius_file" in config[section]:
                    radius_file[basename] = config[section]["radius_file"]
                    break
            if radius_file[basename] is None:
                radius_file[basename] = config["LightconeCommon"]["radius_file"]

            # Get absolute path to radius file
            if not os.path.isabs(radius_file[basename]):
                radius_file[basename] = os.path.normpath(os.path.join(sim_dir, radius_file[basename]))
            print("Taking shell redshifts for %s from: %s" % (basename, radius_file[basename]))

            # Read in the shell redshifts
            with open(radius_file[basename], "r") as f:
                header = f.readline()
                # Check for the expected header
                if header.strip() != "# Minimum redshift, Maximum redshift":
                    raise Exception("Shell radii must be specified as redshifts!")
                # Read radii
                z_central[basename] = []
                while True:
                    line = f.readline().strip()
                    if len(line) == 0:
                        break
                    else:
                        z_central[basename].append(0.5*sum([float(col) for col in line.split(",")]))

            z_central[basename] = np.asarray(z_central[basename], dtype=float)
            print("  Found %d shell redshifts" % len(z_central[basename]))
    
        # Accumulate correct_shell_file() arguments needed to process each shell
        args = []
        for basename in basenames:

            # Find number of shells in this lightcone
            with h5py.File("%s/%s_index.hdf5" % (indir, basename), "r") as infile:
                nr_shells = int(infile["Lightcone"].attrs["nr_shells"])
                nr_files_per_shell = int(infile["Lightcone"].attrs["nr_files_per_shell"])
            if nr_files_per_shell != 1:
                raise Exception("Can only correct single file outputs!")
            if nr_shells != len(z_central[basename]):
                raise Exception("Wrong number of shell redshifts!")

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
            args += [(indir, basename, outdir, shell_nr, z_central[basename], neutrino_density)
                     for shell_nr in range(nr_shells)]

    # Make sure all output dirs have been created
    comm.barrier()

    # Apply corrections to all shells in parallel
    with MPICommExecutor(comm, root=0) as executor:
        if executor is not None:
            # Need to fetch results to ensure exception is raised on failure
            results = list(executor.starmap(correct_shell_file, args))


def main():
            
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
        if len(sys.argv) < 5:
            print(usage)
            args = None
        else:
            args["indir"] = sys.argv[1]
            args["yml_file"] = sys.argv[2]
            args["outdir"] = sys.argv[3]
            args["snapfile"] = sys.argv[4]
            args["basenames"] = sys.argv[5:]
    args = comm.bcast(args)

    if args is None:
        MPI.Finalize()
        sys.exit(0)

    correct_maps_mpi(args["indir"], args["yml_file"], args["outdir"], args["snapfile"], args["basenames"])

    comm.barrier()
    if comm_rank == 0:
        print("All shells done for lightcones ", args["basenames"])
