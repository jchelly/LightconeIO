#!/bin/env python

import collections.abc
import os
import hashlib
import numpy as np
import h5py
import unyt

import lightcone_io.units
from lightcone_io.utils import LocalOrRemoteFile


class HealpixMap(collections.abc.Sequence, LocalOrRemoteFile):
    """
    Class used to read lightcone HEALPix maps written out by SWIFT.
    Numpy-style indexing can be used to read pixels from the map.

    :param filenames: names of the HDF5 files containing map data
    :type  filenames: list of str
    :param map_name: name of the map to read
    :type  map_name: str
    :param remote_dir: remote directory containing the file, or None
    :type  remote_dir: hdfstream.RemoteDirectory, or None for local files
    """
    def __init__(self, filenames, map_name, remote_dir=None):
        self.set_directory(remote_dir)
        self._filenames = filenames
        self._map_name  = map_name
        self._nr_pixels = None

    def _set_metadata(self):
        if self._nr_pixels is None:
            with self.open_file(self._filenames[0]) as infile:
                self._nr_pixels = infile[self._map_name].attrs["number_of_pixels"][0]
                self._nside = infile[self._map_name].attrs["nside"][0]
                self._dtype = infile[self._map_name].dtype
                self._pix_per_file = infile[self._map_name].shape[0]
                if unyt is not None:
                    self._units = lightcone_io.units.units_from_attributes(infile[self._map_name])

    @property
    def nside(self):
        """
        Return the nside (resolution) parameter of this HEALPix map
        """
        self._set_metadata()
        return self._nside

    @property
    def dtype(self):
        """
        Return the numpy data type of this HEALPix map
        """
        self._set_metadata()
        return self._dtype

    @property
    def units(self):
        """
        Return the unyt units of this HEALPix map
        """
        self._set_metadata()
        return self._units

    def __len__(self):
        self._set_metadata()
        return self._nr_pixels

    def read_pixels(self, start, stop):
        """
        Read the specified range of pixels from the map and return a numpy
        array.

        :param start: index of the first pixel to read
        :type  start: int
        :param stop: index just after the last pixel to read
        :type  stop: int

        :return: a numpy array with the specified pixel data
        :rtype: numpy.ndarray
        """
        self._set_metadata()

        if stop < start:
            raise KeyError("range of pixels to read must have stop >= start")

        if start < 0 or start >= self._nr_pixels:
            raise KeyError("start of range of pixels to read is out of range")

        if stop < 0 or stop > self._nr_pixels:
            raise KeyError("end of range of pixels to read is out of range")

        # Create output array
        pixels = np.ndarray(stop-start, dtype=self._dtype)

        # Check for zero size output
        if stop == start:
            if unyt is None:
                return pixels
            else:
                return unyt.unyt_array(pixels, self._units)

        # Determine which files to read
        nr_files = len(self._filenames)
        first_file = start // self._pix_per_file
        last_file = (stop-1) // self._pix_per_file
        if last_file >= nr_files:
            last_file = nr_files - 1

        # Loop over files
        output_offset = 0
        for file_nr in range(first_file, last_file+1):

            # Find offset to first pixel in this file
            file_offset = file_nr*self._pix_per_file

            # Open this file
            with self.open_file(self._filenames[file_nr]) as infile:

                # Find number of pixels in the file
                nr_pix_in_file = infile[self._map_name].shape[0]

                # Find range of pixels to read from this file
                i1 = start - file_offset
                if i1 < 0:
                    i1 = 0
                i2 = stop - file_offset
                if i2 > nr_pix_in_file:
                    i2 = nr_pix_in_file

                # Read pixels
                if i2 > i1:
                    nr_to_read = i2 - i1
                    pixels[output_offset:output_offset+nr_to_read] = infile[self._map_name][i1:i2]
                    output_offset += nr_to_read

        assert output_offset == (stop-start)

        if unyt is None:
            return pixels
        else:
            return unyt.unyt_array(pixels, self._units)

    def __getitem__(self, item):
        self._set_metadata()
        if isinstance(item, slice):
            start = item.start
            if start is None:
                start = 0
            if start < 0:
                start = self._nr_pixels + start
            stop = item.stop
            if stop is None:
                stop = self._nr_pixels
            if stop < 0:
                stop = self._nr_pixels + stop
            if stop < start:
                stop = start
            step = item.step
            if step is not None and step != 1:
                raise KeyError("Healpix map slices must be contiguous")
        elif item is Ellipsis:
            start = 0
            stop = self._nr_pixels
        else:
            raise KeyError("HealpixMap index must be range [x:y] or Ellipsis [...]")

        return self.read_pixels(start, stop)


class Shell(collections.abc.Mapping, LocalOrRemoteFile):
    """
    Dict-like container for all of the :class:`HealpixMap` instances
    associated with a lightcone shell. Subscripting a :class:`Shell`
    with a map name returns a :class:`HealpixMap`.

    :param basedir: directory containing the lightcone outputs
    :type  basedir: str
    :param basename: name of the subdirectory for this lightcone (e.g. ``lightcone0``)
    :type  basename: str
    :param shell_nr: index of the lightcone shell to read
    :type  shell_nr: int
    :param remote_dir: remote directory containing the file, or None
    :type  remote_dir: hdfstream.RemoteDirectory, or None for local files
    :param have_shell_dir: True if each shell is in a shell_X subdirectory, False if not, None if not known
    :type have_shell_dir: bool or None
    """
    def __init__(self, basedir, basename, shell_nr, remote_dir=None, have_shell_dir=None):
        self.set_directory(remote_dir)

        # Check if we have a shell_X subdirectory
        if have_shell_dir is None:
            have_shell_dir = self.have_shell_dir(basedir, basename, shell_nr)

        # Find all files that make up this shell
        self.filenames = []
        file_nr = 0
        nr_files = 1
        while file_nr < nr_files:
            fname = self.filename(basedir, basename, shell_nr, file_nr, have_shell_dir)
            self.filenames.append(fname)
            if file_nr == 0:
                with self.open_file(fname) as infile:
                    nr_files = int(infile["Shell"].attrs["nr_files_per_shell"][0])
                    length_unit_cgs = float(infile["Units"].attrs["Unit length in cgs (U_L)"][0])
                    self.comoving_inner_radius = infile["Shell"].attrs["comoving_inner_radius"][0]
                    self.comoving_outer_radius = infile["Shell"].attrs["comoving_outer_radius"][0]
                    if unyt is not None:
                        length_unit = unyt.cm * length_unit_cgs
                        self.comoving_inner_radius = unyt.unyt_quantity(self.comoving_inner_radius, units=length_unit)
                        self.comoving_outer_radius = unyt.unyt_quantity(self.comoving_outer_radius, units=length_unit)
                    self.map_names = []
                    for name in infile:
                        if "nside" in infile[name].attrs:
                            self.map_names.append(name)
            file_nr += 1

        # Create the map objects
        self._maps = {}
        for map_name in self.map_names:
            self._maps[map_name] = HealpixMap(self.filenames, map_name, self._remote_dir)

    @staticmethod
    def filename(basedir, basename, shell_nr, file_nr, have_shell_dir):
        """
        Return the filename for one of the shell files.

        :param basedir: directory containing the lightcone outputs
        :type  basedir: str
        :param basename: name of the subdirectory for this lightcone (e.g. ``lightcone0``)
        :type  basename: str
        :param shell_nr: index of the lightcone shell to read
        :type  shell_nr: int
        :param file_nr: index of the lightcone shell file to read
        :type  file_nr: int
        :param have_shell_dir: True if each shell is in a shell_X subdirectory, False if not
        :type have_shell_dir: bool
        """
        format_with_dir = "{basedir}/{basename}_shells/shell_{shell_nr}/{basename}.shell_{shell_nr}.{file_nr}.hdf5"
        format_without_dir = "{basedir}/{basename}_shells/{basename}.shell_{shell_nr}.{file_nr}.hdf5"
        format_filename = format_with_dir if have_shell_dir else format_without_dir
        return format_filename.format(basedir=basedir, basename=basename, shell_nr=shell_nr, file_nr=file_nr)

    def have_shell_dir(self, basedir, basename, shell_nr):
        filename_with_dir = Shell.filename(basedir, basename, shell_nr, 0, True)
        filename_without_dir = Shell.filename(basedir, basename, shell_nr, 0, False)
        if self.path_exists(filename_without_dir):
            return False
        elif self.path_exists(filename_with_dir):
            return True
        else:
            raise FileNotFoundError("Unable to locate shell file")

    def __getitem__(self, key):
        return self._maps[key]

    def __iter__(self):
        for name in self._maps:
            yield name

    def __len__(self):
        return len(self._maps)


class ShellArray(collections.abc.Sequence, LocalOrRemoteFile):
    """
    Sequence-like container for a set of lightcone shells. This class is
    the recommended way to read lightcone HEALPix maps.

    A :class:`ShellArray` can be indexed with an integer to return a :class:`Shell`.

    :param basedir: directory containing the lightcone outputs
    :type  basedir: str
    :param basename: name of the subdirectory for this lightcone (e.g. ``lightcone0``)
    :type  basename: str
    :param remote_dir: remote directory containing the file, or None
    :type  remote_dir: hdfstream.RemoteDirectory, or None for local files
    """
    def __init__(self, basedir, basename, remote_dir=None):
        self.set_directory(remote_dir)
        self._nr_shells = None # not determined yet
        self._have_shell_dir = None
        self._shell = {} # will cache opened shells
        self.basedir = basedir
        self.basename = basename
        if self.nr_shells == 0:
            raise FileNotFoundError("No lightcone shells found")

    def __getitem__(self, index):
        if index < 0 or index >= self.nr_shells:
            raise IndexError("Shell index is out of range")
        if index not in self._shell:
            self._shell[index] = Shell(self.basedir, self.basename, index, self._remote_dir, self.have_shell_dir)
        return self._shell[index]

    @property
    def have_shell_dir(self):
        """
        Return True if the map files are in shell_X subdirectories.

        :return: True if the shell_0 directory exists, False otherwise
        :rtype: bool
        """
        if self._have_shell_dir is None:
            if self.path_exists(f"{self.basedir}/{self.basename}_shells/shell_0"):
                self._have_shell_dir = True
            else:
                self._have_shell_dir = False
        return self._have_shell_dir

    @property
    def nr_shells(self):
        """
        Return the number of shells in this lightcone output by counting files.

        :return: the number of shells.
        :rtype: int
        """
        if self._nr_shells is None:
            self._nr_shells = 0
            while True:
                shell_file = Shell.filename(self.basedir, self.basename, self.nr_shells, 0, self.have_shell_dir)
                if not self.path_exists(shell_file):
                    break
                self._nr_shells += 1
        return self._nr_shells

    def __len__(self):
        return self.nr_shells

def map_file_name(basedir, basename, shell_nr, file_nr):
    return ("%s/%s_shells/shell_%d/%s.shell_%d.%d.hdf5" %
            (basedir, basename, shell_nr, basename, shell_nr, file_nr))

def combine_healpix_maps(indir, basename, shell_nr, outdir):

    # Create the output file
    outname = map_file_name(outdir, basename, shell_nr, 0)
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    outfile = h5py.File(outname, "w")
    print("Created output file: %s" % outname)

    # Open the first input file
    inname = map_file_name(indir, basename, shell_nr, 0)
    infile = h5py.File(inname, "r")
    print("Opened input file: %s" % inname)

    # Copy metadata
    infile.copy(source="Units", dest=outfile)
    infile.copy(source="InternalCodeUnits", dest=outfile)
    infile.copy(source="Shell", dest=outfile)
    outfile["Shell"].attrs["nr_files_per_shell"] = (1,)

    # Read the input unit information: this has the M, L, T etc units in cgs
    input_units_cgs = lightcone_io.units.read_cgs_units(infile)

    # Get list of datasets
    datasets = []
    for name in infile:
        if "nside" in infile[name].attrs:
            datasets.append(name)
            print("  found dataset %s" % name)

    # Determine total number of pixels
    number_of_pixels = infile[datasets[0]].attrs["number_of_pixels"]
    print("There are %d pixels per map" % number_of_pixels)

    # Create output datasets
    # Note: this will not preserve lossy compression filters
    for name in datasets:
        dset = infile[name]
        shape = (number_of_pixels,)
        outfile.create_dataset(name, shape=shape, dtype=dset.dtype,
                               chunks=dset.chunks, compression=dset.compression,
                               compression_opts=dset.compression_opts,
                               shuffle=dset.shuffle)
        for attr_name in infile[name].attrs:
            outfile[name].attrs[attr_name] = infile[name].attrs[attr_name]
        print("Created output dataset %s" % name)

        # Correct the unit metadata if necessary
        corrections = lightcone_io.units.correct_units(dset, name, input_units_cgs)
        if len(corrections) > 0:
            print("WARNING: modifying unit info for ", name)
        for attr_name, attr_value in corrections.items():
            print("  Setting attribute ", attr_name, " to ", attr_value)
            outfile[name].attrs[attr_name] = attr_value

    # Close the input file
    infile.close()

    # Copy data. Here we need to loop over the input files.
    offset = 0
    file_nr = 0
    while offset < number_of_pixels:

        # Open the next input file
        inname = map_file_name(indir, basename, shell_nr, file_nr)
        infile = h5py.File(inname, "r")
        print("Opened input file to copy pixel data: %s" % inname)

        # Copy pixel data from this file
        local_pixels = None
        for name in datasets:
            dset = infile[name]

            # Check number of pixels in this file
            if local_pixels is None:
                local_pixels = dset.shape[0]
            else:
                if local_pixels != dset.shape[0]:
                    raise Exception("All maps must be the same size!")

            # Copy the pixel data for this map
            outfile[name][offset:offset+local_pixels] = infile[name][:]

        # Next file
        offset += local_pixels
        file_nr += 1

    if offset != number_of_pixels:
        raise Exception("Number of pixels copied is wrong!")

    print("Finished merging files")


def fetch_pixels(indir, basename, shell_nr, dataset):

    max_per_fetch = 10*1024*1024

    # Loop over input files
    file_nr = 0
    total_read = 0
    while True:

        # Open the current file and dataset
        inname = map_file_name(indir, basename, shell_nr, file_nr)
        infile = h5py.File(inname, "r")
        dset = infile[dataset]
        if file_nr == 0:
            total_pixels = dset.attrs["number_of_pixels"]

        # Loop over and return pixels in this file
        for offset in range(0, dset.shape[0], max_per_fetch):
            i1 = offset
            i2 = offset + max_per_fetch
            if i2 > dset.shape[0]:
                i2 = dset.shape[0]
            assert (i2 > i1)
            yield dset[i1:i2]
            total_read += (i2 - i1)

        # Next file
        infile.close()
        file_nr += 1

        # Check if we're done
        assert total_read <= total_pixels
        if total_read == total_pixels:
            break


def compare_healpix_maps(indir1, indir2, basename, shell_nr, dataset):

    # Compute hash of first map
    map1 = fetch_pixels(indir1, basename, shell_nr, dataset)
    map1_hash = hashlib.sha256()
    for pixel_data in map1:
        map1_hash.update(pixel_data)
    map1_hash = map1_hash.digest()

    # Compute hash of second map
    map2 = fetch_pixels(indir2, basename, shell_nr, dataset)
    map2_hash = hashlib.sha256()
    for pixel_data in map2:
        map2_hash.update(pixel_data)
    map2_hash = map2_hash.digest()

    return (map1_hash == map2_hash)


def get_map_names(fname):
    map_names = []
    with h5py.File(fname, "r") as infile:
        for name in infile:
            if isinstance(infile[name], h5py.Dataset):
                map_names.append(name)
    return map_names


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
        n = len(datasets)
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
