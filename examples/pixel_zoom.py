import sys
import numpy as np
import healpy as hp
import h5py
import lightcone_io.healpix_maps as hm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl

def fetch_zoom_pixels(filename, centre_pixel_idx, pixel_idx, nside, map_name):
    """
        Retrieve selected pixels from a map to make a zoomed in 
        gnomview plot of a region on the sky.

        filename: path to hdf5 file of the map or lightcone_io healpixmap shell object
        centre_pix_idx: index of the pixel to centre the map on.
        pixel_idx: The indices of the pixels to include in the map.
        nside: the nside resolution of the map. 
        map_name: name of the maps dataset within the file

        Returns:
            An empty map aside from the selected pixels. 
    """
    #make empty map:
    xmap = np.zeros(hp.nside2npix(nside))
    if isinstance(pixel_idx, list):
        pixel_idx = np.asarray(pixel_idx).dtype(int)
    
    if isinstance(filename, str):
        # assume string implies that the map is hdf5 object 
        with h5py.File(filename, 'r') as f:
            if pixel_idx is None:
                xmap += f[map_name][:]
                centre_pix_val = f[map_name][centre_pixel_idx]
                
            else:
                pixel_idx=pixel_idx[np.argsort(pixel_idx)].astype(int) # order and ensure correct type
                xmap[pixel_idx]+=f[map_name][pixel_idx]
    
                centre_pix_val = f[map_name][centre_pixel_idx]

    elif isinstance(filename, hm.Shell):
        #lightconeIO shell has been passed instead of string 
        if pixel_idx is None:
             xmap += filename[map_name][:]
             centre_pix_val = f[map_name][centre_pixel_idx]
        else:
            pixel_idx=pixel_idx[np.argsort(pixel_idx)].astype(int) # order and ensure correct type
            xmap[pixel_idx]+=np.array([filename[map_name][n_idx:n_idx+1][0] for n_idx in pixel_idx])
            centre_pix_val = filename[map_name][centre_pixel_idx:centre_pixel_idx+1][0]
             
    return xmap, centre_pix_val


def plot_zoom_on_pixel(filename, nside, centre_pix_idx, map_names, axes_idx=None, output_filename=None, r_npix=10, f_zoom=None, show_plot=False, colormap="cubehelix", bad_colours="grey"):
    """
        Make a gnomview plot of a disk centered on a given pixel. 
        
        filename: path to hdf5 file of the map or lightcone_io healpixmap shell object
        centre_pix_idx: index of the pixel to centre the map on
        map_names: names of the maps to include in the plot
        output_filename: path to output plot. 
        f_zoom: function to apply to selected pixels 
        r_npix: radius of the disk in number of pixels. 
        show_plot: if True show the matplotlib plot object
        axes_idx: dictionary of map names and the subplots row and column indices
        colormap: name of the colour map to use 
        bad_colours: name of the colour be be assigned to bad value or missing pixels
        
    """
    
    theta, phi = hp.pix2ang(nside, centre_pix_idx, lonlat=True) # lonlat=True => [degrees]
    xyz = hp.pix2vec(nside, centre_pix_idx)
    
    pix_sidelength = hp.nside2resol(nside, arcmin=True)*unyt.arcmin
    
    include_pix_idx=hp.query_disc(nside, xyz, r_npix*pix_sidelength.to_value(unyt.radian), inclusive=True)

    if axes_idx is None:
        axes_idx={}
        n = len(map_names)
        if n>3:
            ncols=3
        elif n==4:
            ncols=2
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
    
    cmap = mpl.colormaps.get_cmap(colormap)
    cmap.set_bad(color=bad_colours)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6.5,5), gridspec_kw={'height_ratios': [1,1], 'width_ratios': [1,1,1]})
    
    for map_name in map_names:
        print(f"\n{map_name}")
        xmap, centre_pix_val = fetch_zoom_pixels(filename, nside, centre_pix_idx,  map_name, include_pix_idx)
        
        img_res=hp.nside2resol(nside, arcmin=True) #[arcmin]
    
        img_xy=(2*r_npix+1, 2*r_npix+1)
        
        map_zoom=hp.gnomview(xmap, rot=[theta,phi], xsize=img_xy[0], ysize=img_xy[1], reso=img_res, cmap=None, norm=None, return_projected_map=True, no_plot=True) 
        
        # Sanity check gnom projector
        gnom_obj = hp.projector.GnomonicProj(rot=[theta,phi], xsize=img_xy[0], ysize=img_xy[1], reso=img_res)

        axes_extent = [gnom_obj.get_extent()[0],
                       gnom_obj.get_extent()[1],
                       gnom_obj.get_extent()[2],
                       gnom_obj.get_extent()[3]]
        ax_ij = axes_idx[map_name]
        #print(ax_ij)
        ax=axs[ax_ij[0], ax_ij[1]]

        if f_zoom is None:
            map_zoom_for_plot=map_zoom
        else:
            
            map_zoom_for_plot=f_zoom(map_zoom)

        img = ax.imshow(map_zoom_for_plot, origin='lower', cmap=cmap, norm='log', extent=axes_extent, interpolation="none")
        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)

        if ax_ij[0]==0:
            fig.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, location='top')
        elif ax_ij[0]==1:
            fig.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, location='bottom')


        ## need to add scale bar
        Lx=gnom_obj.get_extent()[1] - gnom_obj.get_extent()[0]
        dx=np.radians((1*unyt.arcmin).to_value(unyt.degree))
        #dx=np.radians(1)
        x0 = gnom_obj.get_extent()[0] + (Lx*0.05)
        x1=x0+dx
        
        y0=gnom_obj.get_extent()[2] + Lx*0.05
        y1=y0
        
        ax.plot([x0,x1], [y0,y1], linewidth=1., color='white')
        ax.text(
            x0+0.5*(x1-x0),
            (y0)+dx/10,
            r"$1$ [arcmin]",
            ha='center',
            va='bottom',
            color='white',
        )
        
        ax.text(
            0.5,
            0.975,
            map_name,
            ha='center',
            va='top',
            color='white',
            transform = ax.transAxes,    
        )
        
    
    plt.savefig(f"{output_filename}", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
