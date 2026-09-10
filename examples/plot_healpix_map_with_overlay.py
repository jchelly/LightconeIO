#!/bin/env python
import sys
import numpy as np
import healpy as hp
import unyt
import math
from lightcone_io.lightcone_vis_tools import add_to_mollweide
import hdfstream
import lightcone_io.healpix_maps as hm
import lightcone_io.halo_reader as hr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



#################################### Example 1 ####################################

def make_example_map(nside = 128, rotation_deg=[-30, -45, 0]):
    
    """
    Make a fun example healpix map from a gaussian beam
    """
    
    m_out = np.zeros(hp.nside2npix(nside)) # output map 

    lmax = int(nside*2)
    #beam_width = 17 * unyt.degree
    beam_width = 30 * unyt.degree
    

    blm=hp.blm_gauss(beam_width.to_value(unyt.radian), lmax=lmax, pol=True) # include polarization to make map more interesting
    m = hp.alm2map(blm, lmax=lmax, mmax=2, nside=nside, pol=True)

    # combine modes to make interesting pattern 
    m = (m[1] * m[2]) - m[0]

    # rotate map so pattern is not at the poles
    rot=hp.Rotator(rot=rotation_deg, inv=True, deg=True, eulertype='ZYX')
    m_rot = rot.rotate_map_alms(m, lmax=lmax)
    
    m_out[m_rot!=0] = 10**(-0.5 * m_rot[m_rot!=0]) # add non_zero vales to output and re-scale to make features more obvious

    return m_out


# make a fun example map at nside 128 and rotate about each axis  
nside=128
rot_deg=[-30, -80, 0]
m=make_example_map(nside, rot_deg)


# set limits of the pixel values included in the maps
pix_percentile = np.percentile(m, [50., 100]) # range of upper 50 percentile of pixels 
pix_min, pix_max = pix_percentile[0], pix_percentile[1]

# create plot 
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7), gridspec_kw={'height_ratios': [1]})

# show all ticks for the plot 
cbar_tick_vals=[1, 1e1, 1e2, 1e3]

hp.newvisufunc.projview(
    m,
    title='', 
    rot=(0,0), # do not rotate the map
    cbar=True, extend="both", # colour bar settings: show colour bar and extend in both directions 
    cbar_ticks=cbar_tick_vals,  #show_tickmarkers=True, 
    norm ='log', # normalise colour map as lognorm 
    cmap='twilight_shifted', min=pix_min, max=pix_max, 
    unit=r'$\log_{10}(\mathrm{pix~value})$', #fontsize={'title':10,'xlabel':10,'ylabel':10,'xtick_label':10,'ytick_label':10, 'cbar_label':10,'cbar_tick_label':10},
    graticule=True, # place lat and long grid onto the projection
    graticule_labels=False, # we will overwrite with new larger labels
    longitude_grid_spacing=60, # spacing of longitude lines
    latitude_grid_spacing=30, # spacing of longitude lines
    hold=True,
    flip="geo",
    sub=111,
)


# beautify the colour bar labels and add minor ticks in log space
fig.canvas.draw()
add_to_mollweide.enforce_colourbar_minor_ticks(ax=fig.axes[-1], scale='log', tick_length=(7,3))

# overlay new graticule labels, with path effects
add_to_mollweide.overlay_new_mollweide_graticule(None, linecolour="none", lon_linecolour="none", lat_labelcolour='silver', lon_labelcolour='silver', lon_linestyle="none", lat_linestyle='none', label_patheffects=[path_effects.withStroke(linewidth=1.2, foreground="k"), path_effects.Normal()])

# overlay window showing a zoomed in region of the map
scalebar=(120, r"$2^{\circ}$", "white")
text_patheff=[path_effects.withStroke(linewidth=1., foreground="black"), path_effects.Normal()]


zoom_vec = hp.ang2vec(rot_deg[0], 90+rot_deg[1], lonlat=True) # location to on healpix map to zoom in on 

window_npix=(240, 240) # inset windows number of pixels 
dx=0.4 # size of window in % of main axes
window_vec=[0.45,0.6,dx,dx*(window_npix[1]/window_npix[0])] # position & size of inset window in main axes

add_to_mollweide.add_zoom_region(fig, m,
    pix_min=pix_min, pix_max=pix_max, 
    zoom_vec=zoom_vec, window_vec=window_vec,
    xy_npix=window_npix, pix_res=2, 
    scale_bar=scalebar,
    window_text="Example: 1", 
    text_colour="white", 
    text_path_effects=text_patheff,
    connector_colour="silver",
    cmap='twilight_shifted'
    )

# add second zoomed window
zoom_vec = hp.ang2vec(rot_deg[0]+4, 90+rot_deg[1] - 9, lonlat=True)
window_npix=(720, 240)
window_vec=[0.1,0.15,1.5*dx,1.5*dx*(window_npix[1]/window_npix[0])]
add_to_mollweide.add_zoom_region(fig, m,
    pix_min=pix_min, pix_max=pix_max, 
    zoom_vec=zoom_vec, window_vec=window_vec,
    xy_npix=window_npix, pix_res=1., 
    scale_bar=scalebar,
    window_text="Example: 2", 
    text_colour="white", 
    text_path_effects=text_patheff,
    connector_colour="silver",
    cmap='twilight_shifted'
    )


plt.savefig("./overlay_example.png", dpi=300, bbox_inches='tight')
plt.close()

#################################### Example 2 ####################################

# Repeat, example 1 but overlay different healpix maps of the fiducial L1_m9 simulation 

# Location of the lightcone output relative to the remote directory
root_dir = hdfstream.open("cosma", "/")
basedir="FLAMINGO/L1_m9/L1_m9/healpix_maps/nside_4096" 
basename="lightcone0" # Which lightcone to read
shell_nr = 1  # Which shell to plot

# Read in shell array 
shell = hm.Shell(basedir, basename, shell_nr, remote_dir=root_dir)

# Show comoving distance to the inner and outer edges of the shell
print("\nComoving inner radius = ", shell.comoving_inner_radius)
print("Comoving outer radius = ", shell.comoving_outer_radius)

# access shell redshifts 
root = hdfstream.open("cosma","/")
txt_file = root["FLAMINGO/L1_m9/L1_m9/shell_redshifts_z3.txt"].open()
shell_redshift = np.loadtxt(txt_file, delimiter=",")
min_z = shell_redshift[shell_nr,0]
max_z = shell_redshift[shell_nr,1]
print(f"Redshift range: {min_z} - {max_z}\n")

map_names = ["TotalMass", "ComptonY"] # Which maps to plot

# Read the map data
map_data = shell[map_names[0]][...].to_value("Msun")
gas_map_data = shell[map_names[1]][...].to_value(unyt.dimensionless)

pix_min, pix_max = map_data[map_data>0].min(), map_data.max()
pix_min, pix_max = 1e10, 5e12

# Create plot 
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7), gridspec_kw={'height_ratios': [1]})

# show all ticks for the plot 
cbar_tick_vals=[1e10, 1e11, 1e12]

hp.newvisufunc.projview(
    map_data,
    title='', 
    rot=(0,0), # do not rotate the map
    cbar=True, extend="both", # colour bar settings: show colour bar and extend in both directions 
    cbar_ticks=cbar_tick_vals,  #show_tickmarkers=True, 
    norm ='log', # normalise colour map as lognorm 
    cmap='cubehelix', min=pix_min, max=pix_max, 
    unit=r'$\mathrm{Total~Mass}~[\mathrm{M}_\odot]$', #fontsize={'title':10,'xlabel':10,'ylabel':10,'xtick_label':10,'ytick_label':10, 'cbar_label':10,'cbar_tick_label':10},
    graticule=False, # place lat and long grid onto the projection
    hold=True,
    flip="geo",
    sub=111,
)
# beautify the colour bar labels and add minor ticks in log space
fig.canvas.draw()
add_to_mollweide.enforce_colourbar_minor_ticks(ax=fig.axes[-1], scale='log', tick_length=(7,3))

# select a massive halo in the lightcone and overlay a window zoomed in on it
# Load halo lightcone catalogue
snapshot_nr=76
filename = f"FLAMINGO/L1_m9/L1_m9/halo_lightcone/lightcone0/lightcone_halos_{snapshot_nr:04d}.hdf5"
soap_filename = f"FLAMINGO/L1_m9/L1_m9/SOAP-HBT/halo_properties_{snapshot_nr:04d}.hdf5"
halos = hr.HaloLightconeFile(filename=filename, soap_filename=soap_filename, remote_dir=root_dir)

# List of halo properties to read
properties = ("Lightcone/Redshift", "Lightcone/HaloCentre", "SO/200_crit/TotalMass")

# Read the data
halo_props = halos.read_halos(properties)

# Select most massive halo by M200c
z_mask = (min_z<=halo_props["Lightcone/Redshift"]) & (max_z>=halo_props["Lightcone/Redshift"])
halo_idx = np.argmax(halo_props["SO/200_crit/TotalMass"][z_mask])
# selcted halo properties
halo_M200c = halo_props["SO/200_crit/TotalMass"][z_mask][halo_idx].to_value("Msun")
halo_redshift = halo_props["Lightcone/Redshift"][z_mask][halo_idx].value
halo_vec = halo_props["Lightcone/HaloCentre"][z_mask][halo_idx].to_value("Mpc")


# overlay window showing a zoomed in region of the map
scalebar=(120, r"$2^{\circ}$", "white")
text_patheff=[path_effects.withStroke(linewidth=1., foreground="black"), path_effects.Normal()]

window_npix=(360, 360) # inset windows number of pixels 
dx=0.5 # size of window in % of main axes
window_vec=[0.675,0.05,dx,dx*(window_npix[1]/window_npix[0])] # position & size of inset window in main axes

# text added to overlay 
mass_order = math.floor(math.log10(abs(halo_M200c)))
#window_txt = r"$M_{200\mathrm{c}} =\,$"+f"{halo_M200c/(10**mass_order):.1f}"+r"$\mathrm{M}_\odot$"
window_txt = fr"$M_{{200\mathrm{{c}}}} =\,{halo_M200c/(10**mass_order):.1f}\times10^{{{mass_order:d}}}\,[\mathrm{{M}}_\odot]$"
compton_Y_percentiles = np.percentile(gas_map_data, [30.,100.])
zoom_pix_min, zoom_pix_max = compton_Y_percentiles[0], compton_Y_percentiles[1]
img, axins = add_to_mollweide.add_zoom_region(fig, gas_map_data,
    pix_min=zoom_pix_min, pix_max=zoom_pix_max, 
    zoom_vec=halo_vec, window_vec=window_vec,
    xy_npix=window_npix, pix_res=1, 
    scale_bar=scalebar,
    window_text=window_txt, 
    text_colour="white", 
    text_path_effects=text_patheff,
    text_size=9,
    connector_colour="k",
    cmap='magma',
    return_img=True
    )

# add colour bar to the new plot as well
cax = inset_axes(axins,
                 width="100%",      # 5% of inset width
                 height="7%",   # same height as inset
                 loc="lower center",
                 bbox_to_anchor=(0, -0.1, 1, 1),
                 bbox_transform=axins.transAxes,
                 borderpad=0)
cbar=plt.colorbar(img, cax=cax, orientation="horizontal", extend="both")
cbar.set_label(r"$\mathrm{Compton}-\mathit{y}$", color="k") #, path_effects=text_patheff)
cbar.ax.tick_params(colors="k")

plt.savefig("./overlay_example2.png", dpi=300, bbox_inches='tight')
plt.close()
