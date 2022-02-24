#!/bin/env python
#
# This script plots the particles in a pencil beam from a lightcone
#

import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np

import lightcone_io.particle_reader as pr

# Specify one file from the spatially indexed lightcone particle data
input_filename = "/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/L2800N5040/HYDRO_FIDUCIAL/indexed_lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"

# Part of the sky to plot
vector = (1.0, 0.0, 0.0)  # Vector pointing at a spot on the sky
radius = np.radians(2.0) # Angular radius around that spot

# Redshift range to plot (set to None for all redshifts in the lightcone)
redshift_range = None

# Open the lightcone
lightcone = pr.IndexedLightcone(input_filename)

plt.figure(figsize=(8,8))

# Loop over particle types
ax = None
nr_types = len(lightcone)
xmax_all = 0
for type_nr, ptype in enumerate(lightcone):
    
    print("Particle type ", ptype)

    # Read in the particle positions and masses
    mass_name = "DynamicalMasses" if ptype=="BH" else "Masses"
    property_names = ("Coordinates", mass_name)
    data = lightcone[ptype].read(property_names=property_names,
                                 redshift_range=redshift_range,
                                 vector=vector, radius=radius,)

    # Unpack positions and masses
    pos  = data["Coordinates"]
    mass = data[mass_name]
    print("Read in %d particles" % pos.shape[0])

    # Find limits for 2D histogram
    xmin = np.amin(pos[:,0])
    xmax = np.amax(pos[:,0])
    ymin = np.amin(pos[:,1])
    ymax = np.amax(pos[:,1])
    aspect=(ymax-ymin)/(xmax-xmin)
    nbins_x = 1000
    nbins_y = int(nbins_x*aspect) # for square-ish pixels
    xmax_all = max(xmax, xmax_all)

    # Make a plot for this particle type
    ax = plt.subplot(nr_types, 1, type_nr+1, sharex=ax, sharey=ax)
    plt.hist2d(pos[:,0], pos[:,1], bins=(nbins_x, nbins_y), weights=mass, norm=col.LogNorm())
    plt.gca().set_aspect("equal")
    plt.xlabel("x [Mpc]")
    plt.ylabel("y [Mpc]")
    plt.title(ptype)

    del pos
    del mass
    del data

plt.xlim(0, xmax_all)
plt.tight_layout()
plt.savefig("pencil_beam.png")
plt.show()
