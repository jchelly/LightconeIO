#!/bin/env python

import collections.abc
import os

import numpy as np
import h5py


try:
    import unyt
except ImportError:
    unyt=None
    raise ImportWarning("Unable to import unyt. Will not return unit info.")
else:
    def units_from_attributes(dset):
        cgs_factor = dset.attrs["Conversion factor to CGS (not including cosmological corrections)"][0]
        U_I = dset.attrs["U_I exponent"][0]
        U_L = dset.attrs["U_L exponent"][0]
        U_M = dset.attrs["U_M exponent"][0]
        U_T = dset.attrs["U_T exponent"][0]
        U_t = dset.attrs["U_t exponent"][0]
        return cgs_factor * (unyt.A**U_I) * (unyt.cm**U_L) * (unyt.g**U_M) * (unyt.K**U_T) * (unyt.s**U_t) 


class HealpixMap(collections.abc.Sequence):

    def __init__(self, filenames, map_name):

        self._filenames = filenames
        self._map_name  = map_name
        self._nr_pixels = None

    def _set_metadata(self):
        if self._nr_pixels is None:
            with h5py.File(self._filenames[0], "r") as infile:
                self._nr_pixels = infile[self._map_name].attrs["number_of_pixels"][0]
                self._nside = infile[self._map_name].attrs["nside"][0]
                self._dtype = infile[self._map_name].dtype
                self._pix_per_file = infile[self._map_name].shape[0]
                if unyt is not None:
                    self._units = units_from_attributes(infile[self._map_name])

    @property
    def nside(self):
        self._set_metadata()
        return self._nside

    @property
    def dtype(self):
        self._set_metadata()
        return self._dtype

    @property
    def units(self):
        self._set_metadata()
        return self._units

    def __len__(self):
        self._set_metadata()
        return self._nr_pixels

    def read_pixels(self, start, stop):
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
            infile = h5py.File(self._filenames[file_nr], "r")
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
            infile.close()

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


class Shell(collections.abc.Mapping):

    def __init__(self, basedir, basename, shell_nr):

        # Find all files that make up this shell
        self.filenames = []
        file_nr = 0
        nr_files = 1
        while file_nr < nr_files:
            fname = ("%s/%s_shells/shell_%d/%s.shell_%d.%d.hdf5" % 
                     (basedir, basename, shell_nr, basename, shell_nr, file_nr))
            self.filenames.append(fname)
            if file_nr == 0:
                with h5py.File(fname, "r") as infile:
                    nr_files = infile["Shell"].attrs["nr_files_per_shell"][0]
                    self.comoving_inner_radius = infile["Shell"].attrs["comoving_inner_radius"][0]
                    self.comoving_outer_radius = infile["Shell"].attrs["comoving_outer_radius"][0]
                    self.map_names = []
                    for name in infile:
                        if "nside" in infile[name].attrs:
                            self.map_names.append(name)
            file_nr += 1

        # Create the map objects
        self._maps = {}
        for map_name in self.map_names:
            self._maps[map_name] = HealpixMap(self.filenames, map_name)

    def __getitem__(self, key):
        return self._maps[key]

    def __iter__(self):
        for name in self._maps:
            yield name

    def __len__(self):
        return len(self._maps)


class ShellArray(collections.abc.Sequence):

    def __init__(self, basedir, basename):

        # Get number of shells from the index file
        fname = basedir+"/"+basename+"_index.hdf5"
        with h5py.File(fname, "r") as infile:
            self.nr_shells = infile["Lightcone"].attrs["nr_files_per_shell"][0]

        # Initialise shells
        self._shell = []
        for shell_nr in range(self.nr_shells):
            self._shell.append(Shell(basedir, basename, shell_nr))
        
    def __getitem__(self, index):
        return self._shell[index]

    def __len__(self):
        return len(self._shell)



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
