#!/bin/env python
#
# This script plots the particles in a pencil beam from a lightcone
#

import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np

import lightcone_io.particle_reader as pr

# Specify one file from the spatially indexed lightcone particle data
input_filename = "/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/DMO_FIDUCIAL/lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"

# Part of the sky to plot
vector = (1.0, 0.0, 0.0)  # Vector pointing at a spot on the sky
radius = np.radians(2.0) # Angular radius around that spot

# Redshift range to plot (set to None for all redshifts in the lightcone)
redshift_range = (0.0, 0.25)

# Open the lightcone
lightcone = pr.IndexedLightcone(input_filename)

# Read in the particle positions and masses
property_names = ("Coordinates", "Masses")
data = lightcone["DM"].read(property_names=property_names,
                            redshift_range=redshift_range,
                            vector=vector, radius=radius,)

# Unpack positions and masses
pos  = data["Coordinates"]
mass = data["Masses"]
print("Read in %d particles" % pos.shape[0])

# Find limits for 2D histogram
xmin = np.amin(pos[:,0])
xmax = np.amax(pos[:,0])
ymin = np.amin(pos[:,1])
ymax = np.amax(pos[:,1])
aspect=(ymax-ymin)/(xmax-xmin)
nbins_x = 1000
nbins_y = int(nbins_x*aspect) # for square-ish pixels

# Make a plot
plt.hist2d(pos[:,0], pos[:,1], bins=(nbins_x, nbins_y), weights=mass, norm=col.LogNorm())
plt.gca().set_aspect("equal")
plt.xlabel("x [Mpc/h]")
plt.ylabel("y [Mpc/h]")
plt.show()
