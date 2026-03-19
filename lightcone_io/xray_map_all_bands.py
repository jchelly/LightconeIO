#!/bin/env python
#
# This script is adapted from lightcone_io.smoothed_map
#
# This script uses the lightcone particle data to create a new HEALPix map.
# Parallelised with MPI.
#
# general
import sys
import os
import healpy as hp
import numpy as np
import h5py
import unyt
from unyt import mp
import datetime
import psutil
import argparse
# lightcone io
import lightcone_io.particle_reader as pr
import lightcone_io.kernel as kernel
import lightcone_io.lc_xray_calculator as Xcalc
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone, Xray_Filter
from lightcone_io.smoothed_map import find_angular_smoothing_length, exchange_particles, distribute_pixels, rotate_particle_coordinates, message, rank_message
# virgo dc
import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5
# mpi
from mpi4py import MPI


comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()
projected_kernel = kernel.ProjectedKernel()

def explode_particle_in_all_bands(nside, 
    part_pos, 
    angular_smoothing_length, 
    part_val_0, part_val_1, part_val_2):
    """
    Same as for smoothed map, but now returns values in 3 seperate bands.

    Given a particle's position vector and angular radius,
    return indexes of the pixels it will update and the values
    to add to the pixels.

    Params:
        part_pos: particle's position vector
        part_val: particle's contribution to the map, [0,1,2]=x-ray bands
        angular_smoothing_length: smoothing length in radians
    
    Returns:
        pix_index: array of indexes of the pixels to update
        pix_val: array of values to add to the pixels
    """

    # Normalize position vector
    part_pos = part_pos / np.sqrt(np.sum(part_pos**2, dtype=float))

    # Find radius containing the pixels to update
    angular_search_radius = angular_smoothing_length*kernel.kernel_gamma
            
    # Get pixel indexes to update
    pix_index = hp.query_disc(nside, part_pos, angular_search_radius)
    assert len(pix_index) >= 1

    # For each pixel, find angle between pixel centre and the particle
    pix_vec_x, pix_vec_y, pix_vec_z = hp.pixelfunc.pix2vec(nside, pix_index)
    dp = part_pos[0]*pix_vec_x + part_pos[1]*pix_vec_y + part_pos[2]*pix_vec_z
    dp = np.clip(dp, a_min=None, a_max=1.0)
    pix_angle = np.arccos(dp)

    del pix_vec_x
    del pix_vec_y
    del pix_vec_z

    # Evaluate the projected kernel for each pixel
    pix_weight = projected_kernel(pix_angle/angular_smoothing_length)
    del pix_angle

    # find number of bands being used:
    assert part_val_0 is not None # must have atleast 1 band given as an input
    
    # remove pixels with a weight of 0 (e.g. pix_angle>=search_radius)
    # can occur due to roudning or problem of single particles not being removed
    in_range = pix_weight>0
    n_in_range=np.sum(in_range)
    
    if n_in_range==len(in_range): 
        # all pixels to update have a weight > 0
        del in_range # mask not needed
        pix_weight=pix_weight
        pix_index=pix_index
        pix_weight_total =  np.sum(pix_weight, dtype=float)
        
        pix_val_0 = part_val_0 * pix_weight / pix_weight_total
        pix_val_1 = part_val_1 * pix_weight / pix_weight_total
        pix_val_2 = part_val_2 * pix_weight / pix_weight_total
    
    elif n_in_range<len(in_range) and n_in_range>0: 
        # skip pixels if weight == 0 
        pix_weight=pix_weight[in_range] #update pixel weights
        pix_index=pix_index[in_range] #update pixel index
        del in_range # mask no longer needed 

        pix_weight_total =  np.sum(pix_weight, dtype=float) #weights total
        
        pix_val_0 = part_val_0 * pix_weight / pix_weight_total
        pix_val_1 = part_val_1 * pix_weight / pix_weight_total
        pix_val_2 = part_val_2 * pix_weight / pix_weight_total
    
    else: #there are no pixels to update
        nr_out_of_range=np.sum(in_range==False)
        del in_range
        pix_index=None
        pix_val_0=None
        pix_val_1=None
        pix_val_2=None

    return pix_index, pix_val_0, pix_val_1, pix_val_2


def get_map_attributes(nside, units):
    
    """
    Given a unyt.Unit object, generate SWIFT dataset attributes

    units: the Unit object

    Returns a dict with the maps attributes
    """
    attrs = {}
    # Get CGS conversion factor. Note that this is the conversion to physical units,
    # because unyt multiplies out the dimensionless a factor.
    cgs_factor, offset = units.get_conversion_factor(units.get_cgs_equivalent())

    # values are physical and therefore we do not care about the scale factor
    physical=True
    a_exponent_in_units=0
    a_val=1
    a_exponent=None

    # Ignore h exponenent in map output
    h_exponent=0
    h_val=1.
    
    # Check a_exponent is consistent
    if a_exponent is None:
        assert physical
    else:
        if physical:
            assert a_exponent_in_units == 0
        else:
            assert float(a_exponent_in_units) == a_exponent
    # Something has gone wrong if we have h factors
    assert h_exponent == 0

    # Find the power associated with each dimension
    powers = units.get_mks_equivalent().dimensions.as_powers_dict()
    # Set the attributes
    attrs["Conversion factor to CGS (not including cosmological corrections)"] = [
        float(cgs_factor / (a_val**a_exponent_in_units) / (h_val**h_exponent))
    ]
    attrs["Conversion factor to physical CGS (including cosmological corrections)"] = [
        float(cgs_factor)
    ]
    attrs["U_I exponent"] = [float(powers[unyt.dimensions.current_mks])]
    attrs["U_L exponent"] = [float(powers[unyt.dimensions.length])]
    attrs["U_M exponent"] = [float(powers[unyt.dimensions.mass])]
    attrs["U_T exponent"] = [float(powers[unyt.dimensions.temperature])]
    attrs["U_t exponent"] = [float(powers[unyt.dimensions.time])]
    attrs["h-scale exponent"] = [float(h_exponent)]
    attrs["a-scale exponent"] = [0.0 if a_exponent is None else a_exponent]
    attrs["Value stored as physical"] = [1 if physical else 0]
    attrs["Property can be converted to comoving"] = [0 if a_exponent is None else 1]
    attrs["nside"]=[int(nside)]
    attrs["number_of_pixels"]=[int(hp.nside2npix(nside))]
    attrs["pixel_ordering_scheme"]=bytes('ring',"utf-8")
    attrs["Expression for physical CGS units"]=[str(units.get_cgs_equivalent())]
    attrs["comoving_inner_radius"]=[-1]
    attrs["comoving_outer_radius"]=[-1]

    return attrs


def get_xray_table(observation_bands, table_filename):
    table_dict={}
    with h5py.File(table_filename, 'r') as f:
        table_dict['Bins']={}
        for k, v in f['Bins'].items():
            if k =='Missing_element':
                table_dict['Bins'][k]=v[()]
            else:
                table_dict['Bins'][k]=v[()].astype(np.float32)
        for observation_band_name in observation_bands:
            table_dict[observation_band_name]={}
            for k, v in f[observation_band_name].items():
                table_dict[observation_band_name][k]=v[()].astype(np.float64)
    return table_dict


def get_cosmology_object(snapshot_root_dir):
    """
    Return the cosmology object 
    """
    return Snapshot_Cosmology_For_Lightcone(snapshot_root_dir)



def write_and_restart_output_shell(output_filename, new_map_names, xray_table_filename, input_filename):
    """
    Make sure there is a .hdf5 file to write new X-ray maps into. 

        - If the outputfile .hdf5 file does not exist then create new output file.
        - If outputfile does exist, remove X-ray maps with the same name as those 
        being made and record the restart date. 
    
    
    !! Should only be written with a single rank !!

    """
    start_date = datetime.datetime.now().strftime("%d-%B-%Y") # track date
    
    try:
        # check if the shell exists, then update with restart info for new maps. 
        if os.path.isfile(output_filename):
            print(f"\nWriting to existing shell: {output_filename}\n", flush=True)
            # we will add to the output file
            with h5py.File(output_filename, "a") as outfile:
                # delete dataset with same name as new maps being made
                for map_name in new_map_names:
                    if map_name in outfile:
                        del outfile[map_name]

                # write Restart information statement
                if "__xrayInfo" not in outfile:
                    NewMapInfo = outfile.create_group("__xrayInfo") #create group that new map info can be written too
                elif "__xrayInfo" in outfile:
                    NewMapInfo = outfile["__xrayInfo"]
                    # clear old attrs if they exist for new map name
                    for map_name in new_map_names:
                        if map_name in NewMapInfo.attrs:
                            del NewMapInfo.attrs[map_name]
                for map_name in new_map_names:
                    # write restart date and emissivity table
                    NewMapInfo.attrs[map_name] = (start_date, xray_table_filename)
        else:
            # write a new shell
            with h5py.File(output_filename, "w") as outfile:
                print(f"\nWriting to empty shell: {output_filename}\n", flush=True)
                NewMapInfo = outfile.create_group("__xrayInfo") # create group that new map info can be written too
                NewMapInfo.attrs["ParticleLightcone"]=input_filename #path to particle lightcones used
                NewMapInfo.attrs["Created"]=start_date #date shell was written
                # write date and emissivity table 
                for map_name in new_map_names:
                            NewMapInfo.attrs[map_name] = (start_date, xray_table_filename)
                ShellInfo=outfile.create_group("Shell") 
                ShellInfo.attrs["nr_files_per_shell"]=1 # should only have 1 file per shell, with multiple maps.
                ShellInfo.attrs["comoving_inner_radius"]=-1 # leave as non-physical for now
                ShellInfo.attrs["comoving_outer_radius"]=-1 # leave as non-physical for now
    except:
        raise FileNotFoundError("Cannot find existing or create new output .hdf5 file")
    
    return None


def write_smoothed_xray_map_in_all_bands(
        input_filename, property_names,
        zmin, zmax, nside, cosmo, xray_tables, output_filename,
        xray_type,
        vector = None, radius = None, theta=0., phi=0.,
    ):
    """
    Compute the X-ray emission from hot gas in the particle lightcone and 
    write a HEALPix map for each X-ray observation band for a given X-ray 
    observation type. 

    input_filename  : name of one file from the particle lightcone output
    property_names  : gas particle properties to read in, e.g., 
                        ("Coordinates","ExpansionFactors","Temperatures")
    zmin            : minimum redshift of gas particles used
    zmax            : maximum redshift of gas particles used
    nside           : HEALPix resolution parameter
    cosmo           : cosmology object 
    xray_tables     : X-ray interpolation table, either as path to 
                        file or dictionary
    output_filename : path to h5file that the maps will be written to
    xray_type       : the X-ray observation type, determines units 
                        of output X-ray flux. 
    theta, phi      : angles to rotate particle positions on the 
                        sky [degree] (must be a unyt quantity)
    """
    
    # Ensure property_names list contains required properties for X-ray maps
    property_names = list(property_names)
    required_particle_properties = [
        "SmoothingLengths", "Coordinates","ExpansionFactors", "Masses", 
        "Densities", "SmoothedElementMassFractions","Temperatures", 
        "StarFormationRates"]
    for prop_name in required_particle_properties:
        if prop_name not in property_names:
            property_names.append(prop_name)

    # Open the lightcone
    lightcone = pr.IndexedLightcone(input_filename, comm=comm)

    # Create an empty HEALPix map, distributed over MPI ranks.
    # Here we assume we can put equal sized chunks of the map on each rank.
    nr_total_pixels, nr_local_pixels, local_offset, theta_boundary = distribute_pixels(comm, nside)
    max_pixrad = hp.pixelfunc.max_pixrad(nside)
    message(f"Total number of pixels = {nr_total_pixels}")
    
    # Will read the full sky within the redshift range
    redshift_range = (zmin, zmax)

    # Determine number of particles to read on this MPI rank
    nr_particles_local = lightcone["Gas"].count_particles(redshift_range=redshift_range,
                                                          vector=vector, radius=radius,)
    nr_particles_total = comm.allreduce(nr_particles_local)
    message(f"Total number of particles in selected cells = {nr_particles_total:.4e}")

    # Read in the particle data
    particle_data = lightcone["Gas"].read_exact(property_names, vector, radius, redshift_range)
    nr_parts_tot = comm.allreduce(particle_data["Coordinates"].shape[0])
    message(f"Read in {nr_parts_tot:.4e} particles")
    
    # remove particles that are not able to give an X-ray value
    # X-ray filter particles: exclude based on SFR, temp, RHP, density
    particle_filter = Xray_Filter(particle_data, particle_data["Coordinates"].shape[0], cosmo=cosmo)
    keep_particles = particle_filter.KEEP_FOR_XRAY
    for prop_name in property_names:
        particle_data[prop_name] = particle_data[prop_name][keep_particles]
    
    nr_parts_tot = comm.allreduce(particle_data["Coordinates"].shape[0])
    message(f"Read in {nr_parts_tot:.4e} particles [Post Filtering]")
    del keep_particles # clean up

    # Find the particle positions and smoothing lengths
    if theta !=0. or phi !=0.:
        message(f"Rotating by (theta, phi): ({theta}, {phi})")
        part_pos_send = rotate_particle_coordinates(particle_data, theta, phi)
    else:
        part_pos_send = particle_data["Coordinates"]

    part_hsml_send = find_angular_smoothing_length(part_pos_send, particle_data["SmoothingLengths"])


    # Determine range of colatitudes each particle will update.
    # Smoothing kernel drops to zero at kernel_gamma * smoothing length.
    # Note that particles with radius < max_pixrad can still update
    # pixels up to max_pixrad away because they update whatever pixel
    # they are in.

    radius = np.maximum(kernel.kernel_gamma*part_hsml_send, max_pixrad) # Might update pixels with centres within this radius
    theta, phi = hp.pixelfunc.vec2ang(part_pos_send)
    part_min_theta = np.clip(theta-radius, 0.0, np.pi) # Minimum central theta of pixels each particle might update
    part_max_theta = np.clip(theta+radius, 0.0, np.pi) # Maximum central theta of pixels each particle might update

    # Determine what range of MPI ranks each particle needs to be sent to
    part_first_rank = np.searchsorted(theta_boundary, part_min_theta, side="left") - 1
    part_first_rank = np.clip(part_first_rank, 0, comm_size-1)
    part_last_rank  = np.searchsorted(theta_boundary, part_max_theta, side="right") - 1
    part_last_rank  = np.clip(part_last_rank, 0, comm_size-1)
    assert np.all(theta_boundary[part_first_rank]  <= part_min_theta)
    assert np.all(theta_boundary[part_last_rank+1] >= part_max_theta)

    # Determine how many ranks each particle needs to be sent to
    nr_copies = part_last_rank - part_first_rank + 1
    assert np.all(nr_copies>=1) and np.all(nr_copies<=comm_size)

    # Duplicate the particles
    nr_parts = part_pos_send.shape[0]
    index = np.repeat(np.arange(nr_parts, dtype=int), nr_copies)
    part_pos_send  = part_pos_send[index,...]
    #part_val_send  = part_val_send[index] # not needed, 3 different values inside loop
    part_hsml_send = part_hsml_send[index]
    #del index # do not remove index as we use it inside the loop

    # Determine destination rank for each particle copy
    nr_parts = np.sum(nr_copies)                # Total number of copied particles
    offset = np.cumsum(nr_copies) - nr_copies   # Offset to first copy of each particle in array of copies
    part_dest = -np.ones(nr_parts, dtype=int) # Destination rank for each copied particle
    for pfr, nrc, off in zip(part_first_rank, nr_copies, offset):
        part_dest[off:off+nrc] = np.arange(pfr, pfr+nrc, dtype=int)
    assert np.all(part_dest >=0) & np.all(part_dest<comm_size)
    message("Computed destination rank(s) for each particle")

    # Tidy up
    del offset
    del part_first_rank
    del part_last_rank
    del nr_copies

    # Copy particles to their destinations
    part_pos_recv, part_hsml_recv = (
        exchange_particles(part_dest, (part_pos_send, part_hsml_send)))

    # Free send buffers
    del part_pos_send
    del part_hsml_send
    #del part_dest # need to keep copying particles within loop
    
    # determine if particle only updates a single pixel
    single_pixel = part_hsml_recv*kernel.kernel_gamma < max_pixrad # correct smoothing effects, includes smoothing kernel
    nr_single_pixel = comm.allreduce(np.sum(single_pixel)) # total number of single pixel updates
    
    # udpdate part_hsml_recv to now only include multi-pixel particles as we have a single pix flag
    # also reduces overhead
    part_hsml_recv=part_hsml_recv[single_pixel==False]
    
    ####################################################################################################
    # Compute X-ray values in all bands. 

    # get arrays of all combinations of observation bands and types     
    all_observation_bands_per_type, all_observation_types_per_band = Xcalc.get_observation_type_per_band(
        unique_observation_types=[xray_type],
        unique_observation_bands=['erosita-high', 'erosita-low','ROSAT']
        )

    XRAY_VALUES, XRAY_MAP_NAMES = Xcalc.particle_xray_values_for_map(
        ['erosita-high', 'erosita-low','ROSAT'],
        [xray_type],
        particle_data,
        xray_tables,
        part_mask=None) # no need for mask, particles have already been removed
    message(particle_data.keys())
    
    del cosmo # no longer needed 
    del all_observation_bands_per_type # no longer needed
    del all_observation_types_per_band

    del particle_data["SmoothedElementMassFractions"]
    del particle_data["Coordinates"]
    del particle_data["Temperatures"]
    del particle_data["Densities"]
    del particle_data["Masses"]

    message("\nComputed X-ray values in all bands")

    for type_idx, observation_type in enumerate([xray_type]):
        message("\nMapping: {obvs_type} [{obvs_type_units}] in the all bands".format(obvs_type=observation_type, obvs_type_units=XRAY_VALUES[0][:,0].units))

        #ensure all 3 bands are returned
        assert XRAY_VALUES[type_idx].shape[-1] == 3 
        
        # confirm that all map units are the same
        assert XRAY_VALUES[type_idx][:,0].units == unyt.unit_object.Unit(Xcalc.xray_map_observation_type_units_cgs[observation_type])

        # store each band individually 
        part_val_send_eROSITA_high = XRAY_VALUES[0][:,0]
        part_val_send_eROSITA_low  = XRAY_VALUES[0][:,1]
        part_val_send_ROSAT        = XRAY_VALUES[0][:,2]
        
        del XRAY_VALUES[0] # clean up & reduce overhead per cycle
        
        # store the totals in each band
        val_total_global_eROSITA_high   = comm.allreduce(np.sum(part_val_send_eROSITA_high, dtype=float))
        val_total_global_eROSITA_low    = comm.allreduce(np.sum(part_val_send_eROSITA_low, dtype=float))
        val_total_global_ROSAT          = comm.allreduce(np.sum(part_val_send_ROSAT, dtype=float))

        # Find units of the quantity which we're mapping
        map_units = part_val_send_eROSITA_high.units # all bands of the same observer type will have the same units

        all_map_units = comm.allgather(map_units)
        for unit in all_map_units:
            if unit != map_units:
                raise RuntimeError("Quantity to map needs to have the same units on all MPI ranks!")


        ####################################################################################################
        # Construct smoothed map in each band

        # Duplicate particle x-ray values
        part_val_send_eROSITA_high = part_val_send_eROSITA_high[index]
        part_val_send_eROSITA_low  = part_val_send_eROSITA_low[index]
        part_val_send_ROSAT        = part_val_send_ROSAT[index]

        # Copy X-ray values to destination
        part_val_recv_eROSITA_high, part_val_recv_eROSITA_low, part_val_recv_ROSAT = (exchange_particles(part_dest, (part_val_send_eROSITA_high, part_val_send_eROSITA_low, part_val_send_ROSAT)))
        # can now remove send x-ray values
        del part_val_send_eROSITA_high
        del part_val_send_eROSITA_low
        del part_val_send_ROSAT

        # Allocate the output map
        map_data_eROSITA_high   = unyt.unyt_array(np.zeros(nr_local_pixels, dtype=float), units=map_units)
        map_data_eROSITA_low    = unyt.unyt_array(np.zeros(nr_local_pixels, dtype=float), units=map_units)
        map_data_ROSAT          = unyt.unyt_array(np.zeros(nr_local_pixels, dtype=float), units=map_units)

        # Will use unit-less views to carry out the map update to minimize overhead
        part_pos_recv_view = part_pos_recv.ndarray_view() # view particle positions

        # view x-ray maps
        #map_view = map_data.ndarray_view()
        map_view_eROSITA_high   = map_data_eROSITA_high.ndarray_view()
        map_view_eROSITA_low    = map_data_eROSITA_low.ndarray_view()
        map_view_ROSAT          = map_data_ROSAT.ndarray_view()

        # view x-sray values
        part_val_recv_view_eROSITA_high = part_val_recv_eROSITA_high.ndarray_view()
        part_val_recv_view_eROSITA_low  = part_val_recv_eROSITA_low.ndarray_view()
        part_val_recv_view_ROSAT        = part_val_recv_ROSAT.ndarray_view()

        # Now each MPI rank has copies of all particles which affect its local
        # pixels. Process any particles which update single pixels.
        
        local_pix_index = hp.pixelfunc.vec2pix(nside, 
                                               part_pos_recv_view[single_pixel, 0],
                                               part_pos_recv_view[single_pixel, 1],
                                               part_pos_recv_view[single_pixel, 2]) - local_offset
        local = (local_pix_index >=0) & (local_pix_index < nr_local_pixels)

        # add single pixel updates for each map
        np.add.at(map_view_eROSITA_high, local_pix_index[local], part_val_recv_view_eROSITA_high[single_pixel][local])
        np.add.at(map_view_eROSITA_low,  local_pix_index[local], part_val_recv_view_eROSITA_low[single_pixel][local])
        np.add.at(map_view_ROSAT,        local_pix_index[local], part_val_recv_view_ROSAT[single_pixel][local])

        del part_pos_recv_view
        del part_val_recv_view_eROSITA_high
        del part_val_recv_view_eROSITA_low
        del part_val_recv_view_ROSAT

        message(f"\nApplied {nr_single_pixel} single pixel updates\nStarting Smoothing and multi-pixel updates")
    
        ## Discard single pixel particles
        nr_parts = np.sum(np.invert(single_pixel))
        nr_parts_tot = comm.allreduce(nr_parts)

        # view particles that update multiple pixels
        part_pos_view  = part_pos_recv[single_pixel==False,:].ndarray_view() # view array where multipixel updates are needed

        # remove single pixel particles from x-ray values and view only the multi-pixel particles
        part_val_view_eROSITA_high = part_val_recv_eROSITA_high[single_pixel==False].ndarray_view()
        part_val_view_eROSITA_low  = part_val_recv_eROSITA_low[single_pixel==False].ndarray_view()
        part_val_view_ROSAT        = part_val_recv_ROSAT[single_pixel==False].ndarray_view()
        
        # sanity check
        assert len(part_hsml_recv)==len(part_pos_view[:, 0])
        
        # track how long it takes to apply multipixel updates 
        if comm_rank==0:
            smoothing_time_0=datetime.datetime.now()
        else:
            smoothing_time_0=None 

        #local_nr_pixels_updated=0
        local_nr_pixels_out_of_range=0
        for part_nr in range(nr_parts):

            # add smoothing filter here; If shell_nr==0 & density or value in pixel is negligable remove this pixel

            # skip particle if there is no value >0 in all X-ray bands
            if (part_val_view_eROSITA_high[part_nr]+part_val_view_eROSITA_low[part_nr]+part_val_view_ROSAT[part_nr])>0 :

                pix_index, pix_val_eROSITA_high, pix_val_eROSITA_low, pix_val_ROSAT = explode_particle_in_all_bands(
                    nside, part_pos_view[part_nr,:], part_hsml_recv[part_nr], 
                    part_val_view_eROSITA_high[part_nr],
                    part_val_view_eROSITA_low[part_nr], 
                    part_val_view_ROSAT[part_nr]
                )

                # if for some there are no pixles to update continue to next particle. 
                if pix_index is None:
                    local_nr_pixels_out_of_range+=1
                    rank_message(f" All searched pixels out of range, n(pixels searched)={-1*pix_val_ROSAT[0]}", comm_rank)
                    continue

                
                local_pix_index = pix_index - local_offset
                local = (local_pix_index >=0) & (local_pix_index < nr_local_pixels)

                # Don't need to use np.add.at here because pixel indexes are unique
                map_view_eROSITA_high[local_pix_index[local]]   += pix_val_eROSITA_high[local]
                map_view_eROSITA_low[local_pix_index[local]]    += pix_val_eROSITA_low[local]
                map_view_ROSAT[local_pix_index[local]]          += pix_val_ROSAT[local]

        del part_pos_view
        del part_val_view_eROSITA_high
        del part_val_view_eROSITA_low
        del part_val_view_ROSAT

        # give total number of multi-pixel updates done, i.e. the number of particles that have been smoothed
        message(f"\nAdded {nr_parts_tot} multi-pixel particles to the map")
        
        if comm_rank==0:
            smoothing_time_1=datetime.datetime.now()
            dt_str = str(smoothing_time_1-smoothing_time_0)
            rank_message("time spent applying multipixel updates = " + dt_str, comm_rank)
        else:
            smoothing_time_1=None 

        del smoothing_time_0 
        del smoothing_time_1

        comm.barrier() # gather at barrier to then give sanity check update

        ####################################################################################################
        # Sanity chcek & Write maps to output .hdf5 file

        # Sanity check:
        # Sum over the map should equal sum of values to be accumulated to the map.
        particle_total_str="\nparticle totals:\n\teROSITA-high: {total0:.5e}\n\teROSITA-low: {total1:.5e}\n\tROSAT: {total2:.5e}\n"
        message(particle_total_str.format(total0=val_total_global_eROSITA_high, total1=val_total_global_eROSITA_low, total2=val_total_global_ROSAT))

        # sum over each map and give post map making total 
        map_sum_eROSITA_high   = comm.allreduce(np.sum(map_data_eROSITA_high))
        map_sum_eROSITA_low    = comm.allreduce(np.sum(map_data_eROSITA_low))
        map_sum_ROSAT          = comm.allreduce(np.sum(map_data_ROSAT))

        map_total_str="\nmap totals:\n\teROSITA-high: {total0:.5e}\n\teROSITA-low: {total1:.5e}\n\tROSAT: {total2:.5e}\n"
        message(map_total_str.format(total0=map_sum_eROSITA_high, total1=map_sum_eROSITA_low, total2=map_sum_ROSAT))

        ratio_total_str="\ntotals ratio (map/particles):\n\teROSITA-high: {total0:.5f}\n\teROSITA-low: {total1:.5f}\n\tROSAT: {total2:.5f}\n"
        message(ratio_total_str.format(total0=map_sum_eROSITA_high/val_total_global_eROSITA_high, total1=map_sum_eROSITA_low/val_total_global_eROSITA_low, total2=map_sum_ROSAT/val_total_global_ROSAT))

        # message ensures all ranks are gathered after making sanity checks so map writing can start. 
        
        # can now delete all additional total values 
        del map_sum_eROSITA_high
        del map_sum_eROSITA_low
        del map_sum_ROSAT

        del val_total_global_eROSITA_high
        del val_total_global_eROSITA_low
        del val_total_global_ROSAT

        
        if comm_rank==0:
            # start tracking the time spent on writing the maps to .hdf5 file
            writing_time_0=datetime.datetime.now()

        # write maps to .hdf5 file
        # messages enforce comm.barrier() between writing outputs. 
        
        # eROSITA-high
        dataset_name=Xcalc.xray_map_names('erosita-high', observation_type)
        message(f"\nwriting dataset ({dataset_name}) to file:\n {output_filename}")
        with h5py.File(output_filename, "a", driver="mpio", comm=comm) as outfile:
            phdf5.collective_write(outfile, dataset_name, map_data_eROSITA_high, comm)
        message(f"\tcompleted writing {dataset_name}")
        del map_data_eROSITA_high
        del map_view_eROSITA_high

        # eROSITA-low
        dataset_name=Xcalc.xray_map_names('erosita-low', observation_type)
        message(f"writing dataset ({dataset_name}) to file:\n {output_filename}")
        with h5py.File(output_filename, "a", driver="mpio", comm=comm) as outfile:
            phdf5.collective_write(outfile, dataset_name, map_data_eROSITA_low, comm)
        message(f"\tcompleted writing {dataset_name}")
        del map_data_eROSITA_low
        del map_view_eROSITA_low

        # ROSAT
        dataset_name=Xcalc.xray_map_names('ROSAT', observation_type)
        message(f"writing dataset ({dataset_name}) to file:\n {output_filename}")
        with h5py.File(output_filename, "a", driver="mpio", comm=comm) as outfile:
            phdf5.collective_write(outfile, dataset_name, map_data_ROSAT, comm)
        message(f"\tcompleted writing datasets {dataset_name}")
        del map_data_ROSAT
        del map_view_ROSAT
        
        # define the map dataset attributes and add to each bands output maps. 
        if comm_rank==0:
            map_attrs=get_map_attributes(nside, map_units) # compute atttrs
            with h5py.File(output_filename, "a") as outfile:
                for band in ['erosita-high', 'erosita-low','ROSAT']: 
                    # load map dataset object
                    dataset_name = Xcalc.xray_map_names(band, observation_type)
                    Xray_map=outfile[dataset_name]
                    # write attrs for a given map
                    for k, v in map_attrs.items():
                        Xray_map.attrs[k]=v
        
            # give update on time spent writing outputs
            writing_time_1=datetime.datetime.now()
            dt_writing_str=str(writing_time_1-writing_time_0)
            rank_message("Writing maps end time: "+str(writing_time_1)+"\nTime Spent Writing: "+dt_writing_str, comm_rank)
            del writing_time_0
            del writing_time_1


    return None


