#!/bin/env python
#
# This script uses the lightcone particle data to create a new HEALPix map
#

import healpy as hp
import numpy as np

import lightcone_io.particle_reader as pr

# Specify one file from the spatially indexed lightcone
filename = "/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/DMO_FIDUCIAL/lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"

# Open the lightcone
lightcone = pr.IndexedLightcone(filename)

# Create an empty HEALPix map
nside = 16384
npix = hp.pixelfunc.nside2npix(nside)
map_data = np.zeros(npix, dtype=float)

# Quantities to read in
property_names = ("Coordinates", "Masses", "ExpansionFactors")

# Redshift range to read in (we may get particles outside this range due to binning)
redshift_range = (0.0, 0.05)

# Find out how many particles there will be
nr_particles_tot = lightcone["DM"].count_particles(redshift_range=redshift_range)
nr_particles_done = 0

# Loop over particles in the lightcone
for particle_chunk in lightcone["DM"].iterate_chunks(property_names=property_names,
                                                     redshift_range=redshift_range,
                                                     max_particles=10*1024*1024):

    # Unpack position, mass and expansion factor arrays for this chunk
    pos = particle_chunk["Coordinates"]
    m   = particle_chunk["Masses"]
    z   = 1.0/particle_chunk["ExpansionFactors"]-1.0

    print("Read in %d particles" % m.shape[0])

    # Filter out particles outside the required redshift range
    keep = (z >= redshift_range[0]) & (z < redshift_range[1])
    pos = pos[keep,:]
    m = m[keep]
    del z

    # Determine which pixels these particles belong to
    pos = pos.value     # vec2pix can't handle unyt arrays
    pixel = hp.pixelfunc.vec2pix(nside, pos[:,0], pos[:,1], pos[:,2])
    
    # Accumulate pixels to the map
    np.add.at(map_data, pixel, m.value)

    # This way is much slower because it updates all pixels on each iteration
    #map_data += np.bincount(pixel, minlength=npix, weights=m)

    # Report progress
    nr_particles_done += pos.shape[0]
    print("Percentage of particles processed = %.2f" % ((nr_particles_done/nr_particles_tot)*100.0))
