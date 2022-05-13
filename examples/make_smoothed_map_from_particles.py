#!/bin/env python
#
# This script uses the lightcone particle data to create a new HEALPix map
#

import healpy as hp
import numpy as np
import h5py

import lightcone_io.particle_reader as pr
import lightcone_io.kernel as kernel


def update_map(nside, projected_kernel, max_pixrad,
               map_data, pos, value, hsml):
    """
    Accumulate the provided particles to the HEALPix map
    """

    # Loop over particles
    nr_part = pos.shape[0]
    for i in range(nr_part):

        if i % 1000 == 0:
            print(i)
        
        # Find angular size of this particle
        dist = np.sqrt(np.sum(pos[i,:]**2))
        if dist > 10.0*hsml[i]:
            angular_smoothing_length = hsml[i]/dist
        else:
            angular_smoothing_length = np.arctan(hsml[i] / dist)
        angular_search_radius = angular_smoothing_length*kernel.kernel_gamma

        if angular_smoothing_length < max_pixrad:
            # This particle is small and just updates one pixel
            ipix = hp.pixelfunc.vec2pix(nside, pos[i,0].value, pos[i,1].value, pos[i,2].value)
            map_data[ipix] += value[i].value
        else:
            # This particle updates multiple pixels.
            # Find pixels to be updated by this particle
            ipix = hp.query_disc(nside, pos.value[i,:], angular_search_radius)

            # For each pixel, find angle between pixel centre and the particle
            part_vec = pos[i,:]
            pix_vec  = np.column_stack(hp.pixelfunc.pix2vec(nside, ipix))
            dp = np.sum(part_vec[None,:]*pix_vec, axis=1)
            dp[dp > 1.0] = 1.0 # In case of rounding error
            pix_angle = np.arccos(dp)

            # Evaluate the projected kernel for each pixel
            pix_weight = projected_kernel(pix_angle/angular_smoothing_length)

            # Normalize weights so that sum is one
            pix_weight = pix_weight / np.sum(pix_weight)

            # Compute value to add to each pixel (mass, in this case)
            pix_value = pix_weight*value[i]

            # Accumulate contributions to the map
            np.add.at(map_data, ipix, pix_value.value)


# Specify one file from the spatially indexed lightcone particle data
input_filename = "/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"

# Where to write the new HEALPix map
output_filename = "/cosma8/data/dp004/jch/lightcone_map.hdf5"

# Open the lightcone
lightcone = pr.IndexedLightcone(input_filename)

# Create an empty HEALPix map
#nside = 16384
nside = 512
npix = hp.pixelfunc.nside2npix(nside)
map_data = np.zeros(npix, dtype=float)

# Quantities to read in
property_names = ("Coordinates", "Masses", "ExpansionFactors", "SmoothingLengths")

# Redshift range to read in (we may get particles outside this range due to binning)
redshift_range = (0.0006815781, 0.05)

# Part of the sky to do - set vector and radius to None for full sky.
# The pixel values outside this patch will be zero in the output map.
vector = (1.0, 0.0, 0.0)  # Vector pointing at a spot on the sky
radius = np.radians(10.0) # Angular radius around that spot

# Find out how many particles there will be
nr_particles_tot = lightcone["Gas"].count_particles(redshift_range=redshift_range,
                                                    vector=vector, radius=radius,)
nr_particles_read = 0
nr_particles_added = 0
mass_added = 0.0

print("Total particles to read = %d" % nr_particles_tot)


# Tabulate the projected kernel function
projected_kernel = kernel.ProjectedKernel()
max_pixrad = hp.pixelfunc.max_pixrad(nside)


# Loop over particles in the lightcone
for particle_chunk in lightcone["Gas"].iterate_chunks(property_names=property_names,
                                                      redshift_range=redshift_range,
                                                      vector=vector, radius=radius,
                                                      max_particles=10*1024*1024):

    # Unpack position, mass and expansion factor arrays for this chunk
    pos  = particle_chunk["Coordinates"]
    m    = particle_chunk["Masses"]
    z    = 1.0/particle_chunk["ExpansionFactors"]-1.0
    hsml = particle_chunk["SmoothingLengths"]
    nr_particles_read += m.shape[0]

    # Filter out particles outside the required redshift range
    keep = (z >= redshift_range[0]) & (z < redshift_range[1])
    pos = pos[keep,:]
    m = m[keep]
    del z

    # Add the particles to the map
    update_map(nside, projected_kernel, max_pixrad, map_data, pos, m, hsml)
    
    # Accumulate number of particles and mass added to the map so far
    nr_particles_added += m.shape[0]
    mass_added += np.sum(m, dtype=float)

    # Report progress
    percent_done = (nr_particles_read/nr_particles_tot)*100.0
    print("Processed %d of %d particles (%.2f%%)" % (nr_particles_read, nr_particles_tot, percent_done))

print("Total number of particles added to map = %d" % nr_particles_added)
print("Total mass added to map = ", mass_added)
print("Sum over map pixels     = ", np.sum(map_data))

# Write out the result
with h5py.File(output_filename, "w") as outfile:
    outfile["MassMap"] = map_data
print("Wrote map to %s" % output_filename)
