
#!/bin/env python
import sys
import glob
import numpy as np
import lightcone_io.particle_reader as pr
import lightcone_io.lc_xray_calculator as Xcalc
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone, Xray_Filter
import unyt
import h5py


# simulation 
sim="L1000N1800/HYDRO_FIDUCIAL"

# Specify one file from the spatially indexed lightcone particle data
lightcone_nr=0
input_filename = f"/cosma8/data/dp004/flamingo/Runs/{sim}/particle_lightcones/lightcone{lightcone_nr}_particles/lightcone{lightcone_nr}_0000.0.hdf5"

# specify the snapshot to use the cosmology information from 
snapshot_nr=77
snapshot_filename = f"/cosma8/data/dp004/flamingo/Runs/{sim}/snapshots/flamingo_{snapshot_nr:04d}/flamingo_{snapshot_nr:04d}.hdf5"


vector = (1.0, 0.0, 0.0)    # Vector pointing at a spot on the sky
radius = np.deg2rad(1.0)  # Angular radius around that spot
redshift_range = (0.001, 0.05) #redshift range to load

# Open the lightcone
lightcone = pr.IndexedLightcone(input_filename)

# define required part type and properties
ptype = "Gas"
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

# read particle data from lightcone
particle_data = lightcone[ptype].read(
    property_names=property_names,
    redshift_range=redshift_range,
    vector=vector, radius=radius
)

nr_parts = len(particle_data["ExpansionFactors"])
print("Read in {n_all}".format(n_all=nr_parts))
# filter out: 
# 1) starforming particles 
# 2) too cold
# 3) particles that have recently been heated by AGN feedback

# get cosmology information 
cosmo=Snapshot_Cosmology_For_Lightcone(snapshot_filename)
particle_filter = Xray_Filter(particle_data, nr_parts, cosmo=cosmo)

keep_particles = particle_filter.KEEP_FOR_XRAY
print("number of X-ray particles / total: {n_xray}/{n_all}".format(n_xray=np.sum(keep_particles), n_all=nr_parts))


# X-ray observation bands and types to compute values for
observation_bands=['erosita-high','erosita-low','ROSAT']
observation_types=['photons_intrinsic', 'photons_convolved', 'energies_intrinsic', 'energies_convolved']

xray_emissivity_table_filename = "/cosma8/data/dp004/flamingo/Tables/Xray/X_Ray_table_combined.hdf5"

Xray_flux, Xray_names = Xcalc.particle_xray_values_for_map(
    observation_bands,
    observation_types,
    particle_data,
    xray_emissivity_table_filename,
    part_mask=keep_particles)

# Sanity check for the output values
print("\nMap Name, Numb>0")
for i in range(len(Xray_names)):
    for j in range(len(Xray_names[i])):
        print(Xray_names[i][j], np.sum(Xray_flux[i][:, j]>0))


# plot particle distributions for each map.


# plot a penicl beam 





