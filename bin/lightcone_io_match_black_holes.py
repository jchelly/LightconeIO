#!/bin/env python

import os
import numpy as np
import h5py

import virgo.util.match as match
import lightcone_io.particle_reader as pr

def vr_filename(basedir, suffix, snap_nr, file_nr):
    return ("%s/VR/catalogue_%04d/vr_catalogue_%04d.%s.%d" % 
            (basedir, snap_nr, snap_nr, suffix, file_nr))

def snap_filename(basedir, snap_nr, file_nr):
    return ("%s/snapshots/flamingo_%04d/flamingo_%04d.%d.hdf5" %
            (basedir, snap_nr, snap_nr, file_nr))

def halfway(z1, z2):
    """
    Return 'half way' point between two redshifts. Will use half
    way in log(a) for now, for no particular reason.
    """
    log_a1 = np.log10(1.0/(1.0+z1))
    log_a2 = np.log10(1.0/(1.0+z2))
    log_a = 0.5*(log_a1+log_a2)
    return 1.0/(10.0**log_a)-1.0

def get_redshift_bins(basedir, first_snap, last_snap):
    """
    Associate a redshift range with each output time.
    For each output return minimum and maximum redshift.
    """
    
    # Get the redshift of each snapshot
    snap_redshift = {}
    for snap_nr in range(first_snap, last_snap+1):
        fname = snap_filename(basedir, snap_nr, 0)
        with h5py.File(fname, "r") as infile:
            snap_redshift[snap_nr] = float(infile["Header"].attrs["Redshift"])

    # Associate a redshift range with each snapshot
    z_min = {}
    z_max = {}
    for snap_nr in range(first_snap, last_snap+1):
        if snap_nr == first_snap:
            z_max[snap_nr] = snap_redshift[snap_nr]
            z_min[snap_nr] = halfway(snap_redshift[snap_nr], snap_redshift[snap_nr+1])
        elif snap_nr == last_snap:
            z_max[snap_nr] = halfway(snap_redshift[snap_nr-1], snap_redshift[snap_nr])
            z_min[snap_nr] = snap_redshift[snap_nr]
        else:
            z_max[snap_nr] = halfway(snap_redshift[snap_nr-1], snap_redshift[snap_nr])
            z_min[snap_nr] = halfway(snap_redshift[snap_nr], snap_redshift[snap_nr+1])
    
    return z_min, z_max

def read_vr_bh_info(basedir, snap_nr):
    """
    Read black hole particle IDs and positions from VR
    """
    data = {}
    filename = vr_filename(basedir, "properties", snap_nr, 0)
    with h5py.File(filename, "r") as infile:
        if infile["Num_of_files"][0] != 1:
            raise Exception("Only implemented for single file VR output!")
        # ID of most bound BH
        ID_bh = infile["ID_mbp_bh"][...]
        nr_bh = len(ID_bh)
        # Position of most bound BH
        pos_bh = np.ndarray((nr_bh,3), dtype=infile["Xcmbp_bh"].dtype)
        pos_bh[:,0] = infile["Xcmbp_bh"][...]
        pos_bh[:,1] = infile["Ycmbp_bh"][...]
        pos_bh[:,2] = infile["Zcmbp_bh"][...]
        # Number of BHs in the group
        n_bh = infile["n_bh"][...]
        # Position of potential minimum
        pos_cminpot = np.ndarray((nr_bh,3), dtype=infile["Xcmbp"].dtype)
        pos_cminpot[:,0] = infile["Xcminpot"][...]
        pos_cminpot[:,1] = infile["Ycminpot"][...]
        pos_cminpot[:,2] = infile["Zcminpot"][...]
        # Halo mass
        mass = infile["Mass_tot"][...]
        # Unit conversion
        length_to_kpc = float(infile["UnitInfo"].attrs["Length_unit_to_kpc"])
        mass_to_solarmass = float(infile["UnitInfo"].attrs["Mass_unit_to_solarmass"])
        comoving_or_physical = int(infile["UnitInfo"].attrs["Comoving_or_Physical"])
        h_val = float(infile["SimulationInfo"].attrs["h_val"])
        scale_factor = float(infile["SimulationInfo"].attrs["ScaleFactor"])
        boxsize = float(infile["SimulationInfo"].attrs["Period"])
        if comoving_or_physical == 0:
            # Physical units, no 1/h. Convert positions to comoving Mpc
            pos_bh *= (length_to_kpc/1000.0/scale_factor)
            pos_cminpot *= (length_to_kpc/1000.0/scale_factor)
            boxsize *= (length_to_kpc/1000.0/scale_factor)
            # Convert masses to solar masses
            mass *= mass_to_solarmass
        else:
            # Comoving 1/h units. Convert positions to comoving Mpc
            pos_bh *= (length_to_kpc/1000.0/h_val)
            pos_cminpot *= (length_to_kpc/1000.0/h_val)
            boxsize *= (length_to_kpc/1000.0/h_val)
            # Convert masses to solar masses
            mass *= (mass_to_solarmass/h_val)

        # Some halos have no black hole
        ID_bh = ID_bh.astype(np.int64)
        ID_bh[n_bh==0] = -1
        
    return ID_bh, pos_bh, pos_cminpot, mass, boxsize


if __name__ == "__main__":

    basedir="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/"
    lightcone_dir="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcones_z_first_nest/"
    lightcone_base="lightcone1"
    first_snap=0
    last_snap=77
    outdir="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcone_halos/"

    # Find redshift range associated with each snapshot
    z_min, z_max = get_redshift_bins(basedir, first_snap, last_snap)

    # Pick a snapshot to do
    snap_nr = 50

    # Read in the black holes from the VR output
    ID_bh, pos_bh, pos_cminpot, mass, boxsize = read_vr_bh_info(basedir, snap_nr)

    # Read in the black holes from the lightcone in this redshift range
    lightcone = pr.IndexedLightcone(lightcone_dir+"/"+lightcone_base+"_particles/"+lightcone_base+"_0000.0.hdf5")
    lc_data = lightcone["BH"].read_exact(("Coordinates", "ParticleIDs", "ExpansionFactor"),
                                         redshift_range=(z_min[snap_nr], z_max[snap_nr]))

    # For each black hole in the lightcone, find index of matching IDs in the VR output.
    # Each VR halo could get matched many times.
    ptr = match.match(lc_data["ParticleIDs"], ID_bh)

    # Each matched BH becomes a halo in the output halo catalogue.
    # Make an array with the array indexes of the matched lightcone BH particles
    matched = (ptr >= 0)
    halo_lc_bh_index = np.arange(len(ptr), dtype=int)[matched]

    # Make an array with the indexes of the corresponding VR halo in the snapshot.
    # These will not be unique due to the periodic replication.
    halo_vr_index = ptr[matched]
    
    # For each VR halo, get a vector from the black hole position to centre of potential
    delta_pos = pos_cminpot - pos_bh

    # The position of the most bound BH in each halo in the lightcone is given by
    # pos_bh[halo_lc_bh_index,:]. Need to add an offset to get the potential minimum.
    # TODO: deal with units properly!
    halo_pos_cminpot = lc_data["Coordinates"][halo_lc_bh_index,:].value + delta_pos[halo_vr_index,:]

    # Get mass and redshift of each halo
    halo_a = lc_data["ExpansionFactor"][halo_lc_bh_index]
    halo_mass = mass[halo_vr_index]

    # Ensure the output directory exists
    outname = "%s/%s/lightcone_halos_%04d.hdf5" % (outdir, lightcone_base, snapnum)
    outdir = os.path.dirname(outname)
    try:
        os.makedirs(outdir)
    except OSError:
        pass
        
    # Write the catalogue for this snapshot
    with h5py.File(outname, "w") as outfile:
        outfile["Xcminpot"]  = halo_pos_cminpot[:,0]
        outfile["Ycminpot"]  = halo_pos_cminpot[:,1]
        outfile["Zcminpot"]  = halo_pos_cminpot[:,2]
        outfile["ID"]        = halo_vr_index
        outfile["ID_mbp_bh"] = ID_bh
        outfile["ExpansionFactor"] = halo_a
        outfile["Mass"]      = halo_mass
