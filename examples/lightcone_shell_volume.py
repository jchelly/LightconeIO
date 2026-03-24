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


# for a given sim, find the comoving volume of a lightcone shell using 
# the redshift values of the outer and inner edges of each shell,
# then make a simple diagram. 


# simulation 
sim_name="HYDRO_FIDUCIAL"
sim="L1000N1800/"+sim_name
# sim base dir
base_dir="/cosma8/data/dp004/flamingo/Runs/{sim}".format(sim="L1000N1800/"+sim_name)


# create astropy cosmology object for simulation using any of the snapshot files
snapshot_dir = "{base_name}/snapshots".format(base_name=base_dir)
cosmo=Snapshot_Cosmology_For_Lightcone(snapshot_dir)

# get comoving radii for each shell 
redshift_filename = "/cosma8/data/dp004/flamingo/Runs/{sim}/shell_redshifts_z3.txt".format(sim="L1000N1800/"+sim_name)
r_shells = cosmo.shell_comoving_raddii(redshift_filename)
z_shells = cosmo.shell_redshifts

# show a diagram for the first 10 shells
from matplotlib.collections import PatchCollection

fig = plt.figure(figsize=(3.21, 2.5))
plot_grid = fig.add_gridspec(nrows=1, ncols=1,  width_ratios=[1], height_ratios=[1])
axs=plot_grid.subplots(sharex='row', sharey='row')

n_shells=10
colours = mpl.colormaps['terrain_r'](np.linspace(0, 1, n_shells+1))
shell_volumes = np.zeros(n_shells)
for shell_nr in range(n_shells):
    print(
    "Shell: {shell_nr:d}\n\tredshift: {zmin:.3f} -> {zmax:.3f}\n\tradii: {rmin:.3e} -> {rmax:.3e}".format(
        shell_nr=shell_nr,
        zmin=z_shells[shell_nr, 0], zmax=z_shells[shell_nr, -1],
        rmin=r_shells[shell_nr, 0], rmax=r_shells[shell_nr, -1]
        )
    )
    v_shell=cosmo.lightcone_volume(r_shells[shell_nr, 0], r_shells[shell_nr, -1], use_redshift=False)
    print("\tcomoving volume: {v_shell:.3e}".format(v_shell=v_shell))
    shell_volumes[shell_nr]=v_shell

    axs.add_artist(plt.Circle((0, 0), r_shells[shell_nr, -1].to_value("Mpc"), fc=colours[int(shell_nr)+1], ec='black', zorder=-100-shell_nr))

axs.scatter(0,0, fc='crimson', ec='black', marker='X', zorder=100, label='observer')

cmap=mpl.colormaps['terrain_r']
bounds = shell_volumes
print(bounds)
norm = mpl.colors.LogNorm(bounds[0],bounds[-1])
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            orientation='vertical', shrink=1., ax=axs, extend="both",
            label=r"shell volume [cMpc$^3$]")

axs.set_xlim(-100, r_shells[n_shells+1, -1])
axs.set_ylim(-0.5*r_shells[n_shells+1, -1].to_value("Mpc") -50, 0.5*r_shells[n_shells+1, -1].to_value("Mpc")+50)

axs.set_xlabel("radius [cMpc]")
axs.set_ylabel("radius [cMpc]")

axs.legend(
    loc='best',
    frameon=True, fancybox=False,
    edgecolor='black',
    borderpad=0.3,borderaxespad=0.7,
    columnspacing=0.75, 
    labelspacing=0.25, handletextpad=0.25,
    handleheight=1, handlelength=1.8,
)

fig_name="shells_{fig_numb}".format(fig_numb=n_shells)
plt.savefig("./"+fig_name+".png",dpi=300, bbox_inches='tight')

plt.close()
