#!/bin/env python
#general
import sys
import os
import numpy as np
import h5py
#lightcone_io 
import lightcone_io.lc_xray_calculator as Xcalc
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone
from lightcone_io.smoothed_map import message, rank_message
import lightcone_io.xray_map_all_bands as Xmap
#virgo dc
import virgo.mpi.parallel_hdf5 as phdf5
#mpi
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()


def get_cosmology_object(snapshot_root_dir):
    """
    Return the cosmology object 
    """
    return Snapshot_Cosmology_For_Lightcone(snapshot_root_dir)


def L1000N1800_Xray_Map_All_Bands(shell_nr, xray_type, output_filename, input_filename, snapshot_base_dir, nside):
    """
    For a given L1000N1800 simulation use the particle lightcone to make a smooth all-sky map of 
    the x-ray emssion from hot gas in each FLAMINGO observation band for the a given X-ray 
    observation type. 

    The FLAMINGO observation bands are: 
        -   eROSITA high (2.3-8.0 keV), 
        -   eROSITA low (0.2-2.3 keV)
        -   ROSAT (0.5-2.0 keV)
    
    The available observation types depend on the X-ray emissivity table, e.g.
        -   photons_intrinsic
        -   photons_convolved
        -   energies_intrinsic
        -   energies_convolved

    Params:
        shell_nr            : the lightcone shell number to reproduce, indicates the redshift range. 
        xray_type           : X-ray observation type. 
        output_filename     : path/to/output/shells.hdf5  the path to the .hdf5 that the new maps will be written into.
        snapshot_base_dir   : path to directories containing a snapshot, the meta data is required for cosmology
        nside               : HEALPix resolution parameter
    """

    # must change shell redshifts for different boxsize and resolution. 
    shell_redshifts="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/shell_redshifts_z3.txt"

    xray_table_filename=Xcalc.COMBINED_XRAY_EMISSIVITY_TABLE_FILENAME
    message("\nUsing emissivity table: {xray_table_filename}")
    
    if comm_rank == 0:
        # Read shell redshifts
        redshifts = np.loadtxt(shell_redshifts, delimiter=",")
        zmin = redshifts[shell_nr,0]
        zmax = redshifts[shell_nr,1]
        print(f"Reproducing shell {shell_nr} with zmin={zmin} and zmax={zmax}, at Nside {nside}\n", flush=True)
        
        # construct snapshot cosmology object 
        cosmo = get_cosmology_object(snapshot_base_dir)
        
        # pre-load X-ray tables in all bands
        print(f"reading table")
        table_dict = Xmap.get_xray_table(['erosita-high', 'erosita-low', 'ROSAT'], xray_table_filename)
        
    else:
        zmin = None
        zmax = None
        cosmo=None
        table_dict=None
    
    zmin, zmax = comm.bcast((zmin, zmax))
    cosmo=comm.bcast(cosmo)
    table_dict=comm.bcast(table_dict)

    # Specify the gas particle properties to read in
    property_names = (
        "Coordinates",
        "Masses", 
        "ExpansionFactors",
        "Densities", 
        "SmoothedElementMassFractions",
        "Temperatures", 
        "LastAGNFeedbackScaleFactors", 
        "StarFormationRates"
        )

    __ = Xmap.write_smoothed_xray_map_in_all_bands(
        input_filename, property_names, zmin, zmax, nside, 
        cosmo=cosmo, xray_tables=table_dict, 
        output_filename=output_filename, 
        xray_type=xray_type,
        vector = None, 
        radius = None, 
        theta=0., phi=0.) 


if __name__ == "__main__":

    if comm_rank == 0:
        argparser = argparse.ArgumentParser()
        argparser.add_argument("shell_nr", type=int, help="min shell number")
        argparser.add_argument("--nside", type=int, help="nside of output maps")
        argparser.add_argument("--lightcone_nr", type=int, help="observer number")
        argparser.add_argument("--simulation_dir", type=str, help="directory for input particle lightcone")
        argparser.add_argument("--output_dir", type=str, help="directory of the output maps")
        argparser.add_argument("--xray_type", type=str, help="the X-ray observation type")
        params = argparser.parse_args()
        
        shell_nr = params.shell_nr
        print(f"\nshell_nr: {shell_nr}", flush=True)

        # input particle lightcone example filename 
        input_filename='{sim_dir}/particle_lightcones/lightcone{lc_nr}_particles/lightcone{lc_nr}_0000.0.hdf5'.format(sim_dir=params.simulation_dir, lc_nr=params.lightcone_nr)
        
        # corresponding example snapshot filename
        #snapshot_filename='{sim_dir}/snapshots/flamingo_0077/flamingo_0077.0.hdf5'.format(sim_dir=params.simulation_dir)
        snapshot_base_dir='{sim_dir}/snapshots'.format(sim_dir=params.simulation_dir)

        # create empty output file with basic attributes
        output_filename = "{output_dir}/lightcone{lc_nr}.shell_{shell_nr}.hdf5".format(output_dir=params.output_dir, lc_nr=params.lightcone_nr, shell_nr=params.shell_nr)

        nside =params.nside
        xray_type=params.xray_type

    else:
        shell_nr=None
        input_filename=None
        output_filename=None
        snapshot_base_dir=None
        nside=None
        xray_type=None
    
    # broad cast required variables
    shell_nr = comm.bcast(shell_nr)
    input_filename=comm.bcast(input_filename)
    output_filename=comm.bcast(output_filename)
    snapshot_base_dir=comm.bcast(snapshot_base_dir)
    nside=comm.bcast(nside)
    xray_type=comm.bcast(xray_type)

    # make output .hdf5 file of the shell if it does not already exist.
    # if it does exist then remove maps with the same name. 
    if comm_rank==0:
        print(output_filename, flush=True)
        new_map_names = [xray_map_names(band, observation_type) for band in ['erosita-high', 'erosita-low','ROSAT']]
        xray_table_filename=Xcalc.COMBINED_XRAY_EMISSIVITY_TABLE_FILENAME
        __ = Xmap.write_and_restart_output_shell(output_filename, new_map_names, xray_table_filename, input_filename)

    message("output .hdf5 file is good, start map making", show_time=True) # precautionary barrier after writing output shell

    # write maps
    L1000N1800_Xray_Map_All_Bands(shell_nr, xray_type, output_filename, input_filename, snapshot_base_dir, nside)

    message("\n\nDONE :) \n\n")


quit()
