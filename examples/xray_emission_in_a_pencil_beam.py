
#!/bin/env python
import sys
import glob
import numpy as np
import lightcone_io.particle_reader as pr
import lightcone_io.lc_xray_calculator as Xcalc
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone, Xray_Filter
import unyt
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

########## plot settings ########## 
import matplotlib as mpl
# plot settings can be removed if needed 
# set plotting params 
# Ticks inside plots; more space devoted to plot.
plt.rcParams["xtick.direction"] ='in'
plt.rcParams["ytick.direction"] ='in'
plt.rcParams["xtick.top"]=True
plt.rcParams["ytick.right"]= True

# axes font settings
mpl.rcParams['axes.labelsize']=10
mpl.rcParams['xtick.labelsize']=9
mpl.rcParams['ytick.labelsize']=9

plt.rcParams["legend.fontsize"]=9
###################################

# simulation 
sim_name="HYDRO_FIDUCIAL"
sim="L1000N1800/"+sim_name
# sim base dir
base_dir="/cosma8/data/dp004/flamingo/Runs/{sim}".format(sim="L1000N1800/"+sim_name)

# Specify one file from the spatially indexed lightcone particle data
lightcone_nr=0
input_filename = "{base_name}/particle_lightcones/lightcone{lc_nr}_particles/lightcone{lc_nr}_0000.0.hdf5".format(base_name=base_dir, lc_nr=lightcone_nr)



vector = (1.0, 0.0, 0.0)    # Vector pointing at a spot on the sky
radius = np.deg2rad(0.5)  # Angular radius around that spot
redshift_range = (0.001, 0.01) #redshift range to load

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
print("\nread {n_all} gas particles".format(n_all=nr_parts))
# filter out: 
# 1) starforming particles 
# 2) too cold
# 3) particles that have recently been heated by AGN feedback

# get cosmology information 
# specify the snapshot to use the cosmology information from 
snapshot_filename = "{base_name}/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.hdf5".format(base_name=base_dir,  snap_nr=77)
cosmo=Snapshot_Cosmology_For_Lightcone(snapshot_filename)
particle_filter = Xray_Filter(particle_data, nr_parts, cosmo=cosmo)

keep_particles = particle_filter.KEEP_FOR_XRAY
print("number of particles with x-ray values / total: {n_xray}/{n_all} ({n_frac:.5f})\n".format(n_xray=np.sum(keep_particles), n_all=nr_parts, n_frac=(nr_parts/np.sum(keep_particles))))


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
for i in range(len(Xray_names)):
    for j in range(len(Xray_names[i])):
        
        map_str="{xray_value_name}\n\tparticle total: {value_total:.4e}".format(xray_value_name=Xray_names[i][j], value_total=np.sum(Xray_flux[i][:, j]))
        value_range_str="\n\trange (min>0, max): ({min_val:.4e}, {max_val:.4e}) [{val_units}]".format(min_val=np.min(Xray_flux[i][:, j][Xray_flux[i][:, j]>0].value), max_val=np.max(Xray_flux[i][:, j].value), val_units=Xray_flux[i][:, j].units)
        summary_str=map_str+value_range_str
        print(summary_str)

# show distribution of particle values


# create large range of bins 
log_bin_min=-50
log_bin_max=1
log_bin_width=0.5
bin_edges=10**np.arange(log_bin_min-log_bin_width/2, log_bin_max+(log_bin_width*1.5), log_bin_width)
midpoints=0.5*(bin_edges[:-1]+bin_edges[1:])

figure_xray_values = [['photons_intrinsic', 'photons_convolved'], ['energies_intrinsic', 'energies_convolved']]
for fig_idx in range(2):
    numb_cols = 2
    fig = plt.figure(figsize=(5, 2.5))
    plot_grid = fig.add_gridspec(nrows=1, ncols=numb_cols, width_ratios=np.ones(numb_cols, dtype=int), height_ratios=[1])
    axs=plot_grid.subplots(sharex='row', sharey='row')

    for col_nr, xray_type in enumerate(figure_xray_values[fig_idx]):
        ax=axs[col_nr]
        for j, xray_band in enumerate(observation_bands):
            n_pdf, __ =  np.histogram(Xray_flux[col_nr+fig_idx][:, j].value, bins=bin_edges, density=True)

            m = n_pdf>0
            ax.plot(midpoints[m], n_pdf[m], label=xray_band, color=f'C{9-j}', zorder=-200)


        ax.set_xscale("log")
        ax.set_yscale("log")

        xray_units=unyt.unit_object.Unit(Xray_flux[col_nr+fig_idx][:, j].units)
        xray_label_str=rf'$[{xray_units.latex_repr}]$'
        ax.set_xlabel(xray_label_str)
        ax.set_title(observation_types[j], fontsize=9)

        ax.legend(
            loc='best',
            frameon=False, fancybox=False,
            borderpad=0.3,borderaxespad=0.7,
            columnspacing=0.75, 
            labelspacing=0.25, handletextpad=0.25,
            handleheight=1, handlelength=1.8,
        )

    axs[0].set_ylabel(r"PDF")

    fig_name="Xray_value_example_{fig_numb}".format(fig_numb=fig_idx)
    plt.savefig("./"+fig_name+".png",dpi=300, bbox_inches='tight')

    plt.close()

