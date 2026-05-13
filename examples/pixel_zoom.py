import sys
import re
import numpy as np
import healpy as hp
import h5py
import unyt
import lightcone_io.healpix_maps as hm
import lightcone_io.halo_reader as hr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib as mpl
import cmasher as cmr
from lightcone_io.units import units_from_attributes

def plot_settings():
    """
    matplotlib settings for this example. 
    """
    # Line Properties
    plt.rcParams["lines.linewidth"]=1.5 
    # Font options
    plt.rcParams["font.size"]=8
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["text.usetex"] = False
    #plt.rcParams["legend.labelspacing"]=0.4
    plt.rcParams["legend.labelspacing"]=0.25
    plt.rcParams["legend.columnspacing"]=0.75
    plt.rcParams["legend.borderpad"]=0.3
    plt.rcParams["legend.borderaxespad"]=0.7
    # Figure layour settings
    plt.rcParams["figure.constrained_layout.use"] =True
    plt.rcParams["figure.constrained_layout.h_pad"] =0.005
    plt.rcParams["figure.constrained_layout.w_pad"] =0.005
    plt.rcParams["figure.constrained_layout.hspace"]=0.005
    plt.rcParams["figure.constrained_layout.wspace"]=0.005
    # axes font settings
    mpl.rcParams['axes.labelsize']=10
    mpl.rcParams['figure.labelsize']=10
    mpl.rcParams['xtick.labelsize']=10
    mpl.rcParams['ytick.labelsize']=10

def assign_shell_number(input_redshifts, redshift_filename, return_bounds=False):
    """
        Return an array of the healpixmap lightcone shell number corresponding to the input redshifts. 
        
        input_redshifts     : redshift to assign a shell number to
        redshift_filename   : path/to/file/with/shell/redshift/ranges.txt
        return_bounds       : if true, return the minimum and maximum redshifts per shell
    """
    input_redshifts = np.asarray(input_redshifts)
    redshifts = np.loadtxt(redshift_filename, delimiter=",")
    particle_shell_nr = np.zeros(len(input_redshifts), dtype=np.int16)
    particle_shell_nr += int(-1)
    
    part_zmin=np.min(input_redshifts)
    part_zmax=np.max(input_redshifts)
    for shell_nr, zval in enumerate(redshifts):
        zmin, zmax= zval[0], zval[1]
        if zmax < part_zmin:
            continue
        elif zmin > part_zmax:
            continue
        else:
            shell_nr_mask = (input_redshifts>zmin) & (input_redshifts<=zmax)
            particle_shell_nr[shell_nr_mask] += int(shell_nr+1)
    if return_bounds is False:
        return particle_shell_nr.astype(np.int32)
    else:
         return particle_shell_nr.astype(np.int32), redshifts[np.min(particle_shell_nr.astype(np.int32)):np.max(particle_shell_nr.astype(np.int32))+1, :]

def fetch_zoom_pixels(filename, centre_pixel_idx, pixel_idx, nside, map_name):
    """
        Retrieve selected pixels from a map to make a zoomed in 
        gnomview plot of a region on the sky.

        filename:           path to hdf5 file of the map or lightcone_io healpixmap shell object
        centre_pix_idx:     index of the pixel to centre the map on.
        pixel_idx:          the indices of the pixels to include in the map.
        nside:              the nside resolution of the map. 
        map_name:           name of the maps dataset within the file

        Returns:
            An empty map aside from the selected pixels. 
    """
    #make empty map:
    
    if isinstance(pixel_idx, list):
        pixel_idx = np.asarray(pixel_idx).dtype(int)
    if isinstance(filename, str):# assume string implies that the map is hdf5 object 
        with h5py.File(filename, 'r') as f:
            # define map array and units
            map_units=units_from_attributes(f[map_name])
            xmap = unyt.unyt_array(np.zeros(hp.nside2npix(nside)), units=map_units)
            if pixel_idx is None:
                xmap += f[map_name][:]
                centre_pix_val = f[map_name][centre_pixel_idx]*map_units
            else:
                pixel_idx=pixel_idx[np.argsort(pixel_idx)].astype(int) # order and ensure correct type
                xmap[pixel_idx]+=f[map_name][pixel_idx]
                centre_pix_val = f[map_name][centre_pixel_idx]*map_units

    elif isinstance(filename, hm.Shell):#lightconeIO shell has been passed instead of string 
        # define map array and units
        xmap = unyt.unyt_array(np.zeros(hp.nside2npix(nside)), units=filename[map_name].units)
        if pixel_idx is None:
            xmap += filename[map_name][:]
            centre_pix_val = f[map_name][centre_pixel_idx]* filename[map_name].units
        else:
            pixel_idx=pixel_idx[np.argsort(pixel_idx)].astype(int) # order and ensure correct type
            xmap[pixel_idx]+=np.array([filename[map_name][n_idx:n_idx+1][0] for n_idx in pixel_idx])
            centre_pix_val = filename[map_name][centre_pixel_idx:centre_pixel_idx+1][0] * filename[map_name].units
    return xmap, centre_pix_val

def get_no_frac_latex(unit_obj):
    """
    return latex expression without frac{}{} from a unyt object 
    """
    # Get standard LaTeX
    latex = unit_obj.latex_repr
    
    cleaned = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'\1/\2', latex)
    return cleaned

def plot_zoom_on_pixel(filename, nside, centre_pix_idx, map_names, 
                        axes_idx=None, output_filename=None, 
                        r_npix=10,
                        f_pixels=None, show_plot=False, 
                        colormap="cubehelix", bad_colours="grey",
                        text_colour="white",
                        length_scale=1*unyt.arcmin,
                        scale_colour="white",
                        ):
    """
        Make a gnomview plot of a disk centered on a given pixel. 
        
        filename:           path to hdf5 file of the map or lightcone_io healpixmap shell object
        centre_pix_idx:     index of the pixel to centre the map on
        map_names:          names of the maps to include in the plot
        output_filename:    path to output plot. 
        f_pixels:           function to apply to selected pixels 
        r_npix:             radius of the disk in number of pixels. 
        show_plot:          if True show the matplotlib plot object
        axes_idx:           dictionary of map names and the subplots row and column indices
        colormap:           name of the colour map to use. 
        bad_colours:        name of the colour be be assigned to bad value or missing pixels
        length_scale:       a unyt.quantity object with the scale to be displayed on the image. 
                                If None then no scale is included
        
    """
    plot_settings()

    # query the map to find all pixels within a radius of the centre pixel. 
    theta, phi = hp.pix2ang(nside, centre_pix_idx, lonlat=True) # lonlat=True => [degrees]
    xyz = hp.pix2vec(nside, centre_pix_idx)
    
    pix_sidelength = hp.nside2resol(nside, arcmin=True)*unyt.arcmin
    
    include_pix_idx=hp.query_disc(nside, xyz, r_npix*pix_sidelength.to_value(unyt.radian), inclusive=True)
    
    if isinstance(colormap, str):
        colormap=[colormap]*len(map_names)

    # make dictionary to link maps to axes location
    if axes_idx is None:
        axes_idx={}
        n = len(map_names)
        if n==4:
            ncols=2
        elif n>4:
            ncols=3
        nrows = int(np.ceil(n / ncols))
        for i, name in enumerate(map_names):
            row = int(i // ncols)
            col = int(i % ncols)
            axes_idx[name] = [row, col]
    else:
        nrows = 1
        ncols = 1
        for k, v in axes_idx.items():
            if v[0]+1 > nrows:
                nrows=v[0]+1
            if v[-1]+1 > ncols:
                ncols=v[-1]+1

    # determine figsize
    if nrows>=3:
        fig_ysize=7
    elif nrows==2:
        fig_ysize=5.5
    else:
        fig_ysize=3.21
    if ncols>=3:
        fig_xsize=7
    elif ncols==2:
        fig_xsize=5
    else:
        fig_xsize = 3.21
    
    fig = plt.figure(figsize=(fig_xsize, fig_ysize))
    plot_grid = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.02, hspace=0.02, width_ratios=[1]*ncols, height_ratios=[1]*nrows)
    axs=plot_grid.subplots(sharex=True, sharey=True)

    
    for map_idx, map_name in enumerate(map_names):
        #print(f"\tPlotting: {map_name}")
        
        # collect pixel values
        xmap, centre_pix_val = fetch_zoom_pixels(filename, centre_pix_idx, include_pix_idx, nside, map_name)
        
        # convert coordinate system 
        xmap.convert_to_base("galactic")
        centre_pix_val.convert_to_base("galactic")

        # update units for Intrinsic X-ray maps
        if xmap.units.dimensions == unyt.dimensions.energy/unyt.dimensions.time/unyt.dimensions.length**2:
            xmap.convert_to_units("erg/s/cm**2")
        elif xmap.units.dimensions == unyt.dimensions.time ** -1 * unyt.dimensions.length**-2:
            xmap.convert_to_units("photon/s/cm**2")
        
        if centre_pix_val.units.dimensions == unyt.dimensions.energy/unyt.dimensions.time/unyt.dimensions.length**2:
            centre_pix_val.convert_to_units("erg/s/cm**2")
        elif centre_pix_val.units.dimensions == unyt.dimensions.time ** -1 * unyt.dimensions.length**-2:
            centre_pix_val.convert_to_units("photon/s/cm**2")
        
        # define image resolution and coordinates
        img_res=hp.nside2resol(nside, arcmin=True) #[arcmin]
        img_xy=(2*r_npix+1, 2*r_npix+1)
        
        # make gnom projection of selected pixels 
        map_zoom=hp.gnomview(xmap.value, rot=[theta,phi], xsize=img_xy[0], ysize=img_xy[1], reso=img_res, cmap=None, norm=None, return_projected_map=True, no_plot=True) 
        
        # Sanity check gnom projector
        gnom_obj = hp.projector.GnomonicProj(rot=[theta,phi], xsize=img_xy[0], ysize=img_xy[1], reso=img_res)

        axes_extent = [gnom_obj.get_extent()[0],
                       gnom_obj.get_extent()[1],
                       gnom_obj.get_extent()[2],
                       gnom_obj.get_extent()[3]]
        ax_ij = axes_idx[map_name]
        
        # select axis
        ax=axs[ax_ij[0], ax_ij[1]]

        # apply function to pixel values
        if f_pixels is None:
            map_zoom_for_plot=map_zoom
        else:
            
            map_zoom_for_plot=f_pixels(map_zoom)

        # set colour map and bad colours in map
        cmap = mpl.colormaps.get_cmap(colormap[map_idx])
        cmap.set_bad(color=bad_colours)

        # create img
        if map_name == "DopplerB":
            map_name="|"+map_name+"|"
            img = ax.imshow(-1*map_zoom_for_plot, origin='lower', cmap=cmap, norm='log', extent=axes_extent, interpolation="none")

        else:
            img = ax.imshow(map_zoom_for_plot, origin='lower', cmap=cmap, norm='log', extent=axes_extent, interpolation="none")
        # remove ticks
        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)

        # add colour bar with maps units 
        if xmap.units==unyt.dimensionless:
            show_units="[dimensionless]"
        else:
            show_units=r"[${unit_label}$]".format(unit_label=get_no_frac_latex(xmap.units))

        if ax_ij[0]==0:
            cbar=fig.colorbar(img, ax=ax, orientation='horizontal', shrink=0.85, location='top',
            label=show_units,pad=0.01,
            )
        elif ax_ij[0]==1:
            cbar=fig.colorbar(img, ax=ax, orientation='horizontal', shrink=0.85, location='bottom',
            label=show_units,pad=0.01,
            )
        cbar.ax.tick_params(labelsize=10) 

        # add scale 
        if length_scale is not None:

            ## need to add scale bar
            Lx=gnom_obj.get_extent()[1] - gnom_obj.get_extent()[0]
            dx=np.radians((length_scale).to_value(unyt.degree))
            #dx=np.radians(1)
            x0 = gnom_obj.get_extent()[0] + (Lx*0.1)
            x1=x0+dx

            y0=gnom_obj.get_extent()[2] + Lx*0.05
            y1=y0

            ax.plot([x0,x1], [y0,y1], linewidth=1., color=scale_colour, path_effects=[pe.withStroke(linewidth=1.3, foreground="black"), pe.Normal()])
            ax.text(
                x0+(0.5*(x1-x0)),
                (y0)+dx/10,
                f"{length_scale.to_value(length_scale.units):.1f} "+r"[${unit_label}$]".format(unit_label=length_scale.units.latex_representation()),
                ha='center',
                va='bottom',
                path_effects=[pe.withStroke(linewidth=1., foreground="black"), pe.Normal()],
                color=scale_colour,
            )
        
        # print maps name on img
        ax.text(
            0.5,
            0.975,
            map_name,
            ha='center',
            va='top',
            color=text_colour,
            path_effects=[pe.withStroke(linewidth=1., foreground="Black"), pe.Normal()],
            transform = ax.transAxes, fontsize=10, 
        )
        
    # save img
    plt.savefig(f"{output_filename}", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


# Example script for plotting only a section of pixels from the healpix map
# We will select a small region centred on a halo in the lightcone. 
# Then select a disk about the halo and make an image based on selecting 
# only pixels within a given radius around the haloes centre. 

# use lightcone0 of L1_m9 fiducial model 
BoxsizeResolution = "L1000N1800"
SimName="HYDRO_FIDUCIAL"
# first find a halo on the sky at low redshift
lightcone_nr=0
snapshot_number=76 # z=0.05
base_dir = "/cosma8/data/dp004/flamingo/Runs/{box_res}/{sim_name}".format(box_res=BoxsizeResolution, sim_name=SimName)
haloes_filename= base_dir+"/sorted_hbt_lightcone_halos/lightcone{lc_nr}/lightcone_halos_{snapshot_number:04d}.hdf5".format(lc_nr=lightcone_nr, snapshot_number=snapshot_number)
soap_filename = base_dir+"/SOAP-HBT/halo_properties_{snapshot_number:04d}.hdf5".format(snapshot_number=snapshot_number)

# read both soap and halo catalogue together 
halos = hr.HaloLightconeFile(filename=haloes_filename, soap_filename=soap_filename)

# List of halo properties to read
properties = ("Lightcone/HaloCentre", "Lightcone/Redshift", "SO/200_crit/TotalMass")

# to speed up the example we will only look at haloes within a small area on the sky
# Line of sight vector specifying a point on the sky
vector = (1.0, 0.0, 0.0)
# Angular radius around this point (in radians)
radius = np.radians(20.0)

# Read the data
halo_props = halos.read_halos_in_radius(vector, radius, properties)

# select a halo with M200c close to 10^14 Msun
haloes_M200c = np.log10(halo_props['SO/200_crit/TotalMass'].to_value('Msun'))
target_log_M200c = 14
selected_halo_idx = np.argmin(np.abs(target_log_M200c - haloes_M200c))
#print(f"{halo_props['SO/200_crit/TotalMass'][selected_halo_idx].to("Msun"):.2e}")

#determine lightcone shell of selected halo
shell_redshifts = "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/shell_redshifts_z3.txt"

shell_numbers = assign_shell_number([halo_props['Lightcone/Redshift'][selected_halo_idx].value], shell_redshifts, return_bounds=False)
shell_nr = shell_numbers[0]

# read out information about the halo
info_str="\nM200c:\t\t{m200:.2e}\nRedshift:\t{z}\nShell numb:\t{shell_numb:d}".format(m200=halo_props['SO/200_crit/TotalMass'][selected_halo_idx].to('Msun'), z=halo_props['Lightcone/Redshift'][selected_halo_idx].value, shell_numb=shell_nr)
print(info_str)

# find pixel in Nside 4096 healpix maps associated with tracer particle for the selcted halo 
ipix_4096 = hp.vec2pix(
    4096, 
    halo_props["Lightcone/HaloCentre"][selected_halo_idx][0].to_value("Mpc"),
    halo_props["Lightcone/HaloCentre"][selected_halo_idx][1].to_value("Mpc"),
    halo_props["Lightcone/HaloCentre"][selected_halo_idx][2].to_value("Mpc"),
)


# plot a a disk with radius of 25 pixels, centred the selected pixel.
r_pixels = 25
lc_base_dir = base_dir+'/{lc_dir}'
shell_4096 = hm.Shell(lc_base_dir.format(lc_dir="neutrino_corrected_maps_downsampled_4096"), '/lightcone{lc_nr}'.format(lc_nr=lightcone_nr), shell_nr)

# Show all X-ray bands for intrinsic observations 
map_names=[
    'XrayErositaHighIntrinsicEnergies', 
    'XrayErositaLowIntrinsicEnergies', 
    'XrayROSATIntrinsicEnergies', 
    'XrayErositaHighIntrinsicPhotons', 
    'XrayErositaLowIntrinsicPhotons', 
    'XrayROSATIntrinsicPhotons'
]
output_filename="./xray_example_zoom_4096.png"
plot_zoom_on_pixel(shell_4096, 4096, ipix_4096, map_names, 
                        axes_idx=None, output_filename=output_filename, r_npix=r_pixels, f_pixels=None, show_plot=False, 
                        colormap="cubehelix", bad_colours="grey",
                        length_scale=10*unyt.arcmin,)

# Repeat for a much higher nside map
# Re-compute index of centre pixel 
ipix_16384 = hp.vec2pix(
    16384, 
    halo_props["Lightcone/HaloCentre"][selected_halo_idx][0].to_value("Mpc"),
    halo_props["Lightcone/HaloCentre"][selected_halo_idx][1].to_value("Mpc"),
    halo_props["Lightcone/HaloCentre"][selected_halo_idx][2].to_value("Mpc"),
)

# Confirm that the new pixel ID is a child pixel of the selected 4096 map's pixel
low_nside=4096
high_nside=16384
level_diff =int(hp.nside2order(high_nside) - hp.nside2order(low_nside))
child_ipix_16384 = hm.get_children_ipix(ipix=ipix_4096, nside=low_nside, levels=level_diff)
assert ipix_16384 in child_ipix_16384

# Create plot
shell_16384 = hm.Shell(lc_base_dir.format(lc_dir="neutrino_corrected_maps"), '/lightcone{lc_nr}'.format(lc_nr=lightcone_nr), shell_nr)

output_filename="./xray_example_zoom_16384.png"
plot_zoom_on_pixel(shell_16384, 16384, ipix_16384, map_names, 
                        axes_idx=None, output_filename=output_filename, r_npix=r_pixels*4, f_pixels=None, show_plot=False, 
                        colormap="cubehelix", bad_colours="grey",
                        length_scale=10*unyt.arcmin,)


# Reapeat, but compare an X-ray map to other maps of gas properties & the (unsmoothed) dark matter mass map
map_names=[
    'XrayErositaLowIntrinsicEnergies', 
    'DopplerB',
    'SmoothedGasMass', 
    'ComptonY', 
    'DM',
    'DarkMatterMass'
]
colour_maps=["cubehelix", "cmr.eclipse", "cmr.ember" ,"plasma", "cmr.lilac", "cmr.cosmic"]

# Include a function to remove pixels below the 10th percentile. 
def selection_function(x):
    p001 = np.percentile(np.abs(x), [10])[0]
    m=np.abs(x)<p001
    x[m]=0.
    return x

output_filename="./multi_properties_example_zoom_4096.png"
plot_zoom_on_pixel(shell_4096, 4096, ipix_4096, map_names, 
                        axes_idx=None, output_filename=output_filename, r_npix=25, f_pixels=selection_function, show_plot=False, 
                        colormap=colour_maps, bad_colours="grey",
                        length_scale=10*unyt.arcmin,)


output_filename="./multi_properties_example_zoom_16384.png"
plot_zoom_on_pixel(shell_16384, 16384, ipix_16384, map_names, 
                        axes_idx=None, output_filename=output_filename, r_npix=r_pixels*4, f_pixels=selection_function, show_plot=False, 
                        colormap=colour_maps, bad_colours="grey",
                        length_scale=10*unyt.arcmin,)
