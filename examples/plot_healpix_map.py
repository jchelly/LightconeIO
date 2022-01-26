#!/bin/env python
#
# This script plots a HEALPix map
#

import matplotlib.pyplot as plt
import healpy as hp
import numpy as np

import lightcone_io.healpix_maps as hm

# Lightcone base directory
basedir = "/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/DMO_FIDUCIAL/lightcones/"

# Lightcone base name
basename = "lightcone0"

# Which shell to plot
shell_nr = 10

# Which map to plot
map_name = "TotalMass"

# Open the lightcone map set
shell = hm.ShellArray(basedir, basename)

print("Comoving inner radius = ", shell[shell_nr].comoving_inner_radius)
print("Comoving outer radius = ", shell[shell_nr].comoving_outer_radius)

# Read the map
map_data = shell[shell_nr][map_name][...]

# Plot the map
hp.mollview(map_data, norm="log")
plt.show()
