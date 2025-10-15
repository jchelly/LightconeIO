#!/bin/env python

import numpy as np
import h5py
import healpy as hp
import unyt

from .particle_reader import merge_cells
from .units import units_from_attributes


class HaloLightconeFile:
    """
    Class used to read a single file from the halo lightcone. Each
    file contains halos which were identified in a single snapshot of
    the simulation. Halos are sorted according to their pixel index in
    a low resolution HEALPix map so that regions on the sky can be
    extracted efficiently.

    :param filename: Name of the halo lightcone file to open
    :type  filename: str
    """
    def __init__(self, filename):
        self._file = h5py.File(filename, "r")
        self._num_halos_per_pixel = self._file["Index/NumHalosPerPixel"][...]
        self._first_halo_in_pixel = self._file["Index/FirstHaloInPixel"][...]
        self._nside = int(self._file["Index"].attrs["nside"])
        self._order = str(self._file["Index"].attrs["order"])
        self._props = None

    @property
    def properties(self):
        """
        Return a list of all halo property names in this file. These are the
        values which are valid to pass to the properties parameter of
        :py:meth:`read_halos_in_pixels` and :py:meth:`read_halos_in_radius`.
        """
        if self._props is None:
            self._props = self._file["Lightcone"].attrs["property_names"]
        return self._props

    def get_pixels_in_radius(self, vector, radius):
        """
        Return indexes of all healpix pixels within radius of vector

        :param vector: direction vector as an array of 3 floats
        :type  vector: numpy.ndarray
        :param radius: angular radius in radians
        :type  radius: float

        :return: a numpy array of ints with the pixel indexes
        :rtype: numpy.ndarray
        """
        if self._order == "nest":
            nest = True
        elif self._order == "ring":
            nest = False
        else:
            raise RuntimeError("Invalid order parameter")
        pixels = hp.query_disc(self._nside, vector, radius, inclusive=True, nest=nest)
        pixels.sort()
        return pixels

    def read_halos_in_pixels(self, pixels, properties):
        """
        Read halos in the specified HEALPix pixels and return the
        requested halo properties in a dict of numpy arrays.

        :param pixels: array of HEALPix pixel indexes to read, or None to read all
        :type  pixels: numpy.ndarray, or None
        :param properties: list of halo properties (i.e. HDF5 dataset names) to read
        :type  properties: list of str

        :return: dict of arrays with the halo properties
        :rtype:  dict of numpy.ndarray
        """

        if pixels is not None:
            # Determine ranges of halos to read
            if np.any(pixels[1:] <= pixels[:-1]):
                raise RuntimeError("Pixel indexes must be unique and ascending!")

            # Discard any selected pixels with no halos
            pixels = pixels[self._num_halos_per_pixel[pixels] > 0]

            # Compute offset to first halo and number of halos per pixel
            offsets = self._first_halo_in_pixel[pixels]
            counts  = self._num_halos_per_pixel[pixels]

            # Merge any consecutive ranges to read
            offsets, counts = merge_cells(offsets, counts)

        else:
            # We're reading all of the halos, so we just have one range to read
            offsets = np.asarray((0,), dtype=int)
            counts = np.asarray((np.sum(self._num_halos_per_pixel),), dtype=int)

        # Compute expected number of halos
        nr_halos = sum(counts)

        # Loop over halo properties to read
        result = {}
        for name in properties:

            # Locate the dataset with this property
            dataset = self._file[name]

            # Determine output units for this dataset
            units = units_from_attributes(dataset)

            # Allocate the output array
            shape = [nr_halos,]+list(dataset.shape[1:])
            data = unyt.unyt_array(np.ndarray(shape, dtype=dataset.dtype), units)

            # Read data for the selected pixels
            i = 0
            for offset, count in zip(offsets, counts):
                data[i:i+count,...] = dataset[offset:offset+count,...]
                i += count
            assert i == nr_halos
            result[name] = data

        return result

    def read_halos_in_radius(self, vector, radius, properties):
        """
        Read halos in an angular radius around the specified line of sight
        vector and return the requested halo properties in a dict of numpy
        arrays. May also return some halos slightly outside the radius
        because if we read a pixel we read all of the halos in it.

        :param vector: direction vector as an array of 3 floats
        :type  vector: numpy.ndarray
        :param radius: angular radius in radians
        :type  radius: float
        :param properties: list of halo properties (i.e. HDF5 dataset names) to read
        :type  properties: list of str

        :return: dict of arrays with the halo properties
        :rtype:  dict of numpy.ndarray
        """
        pixels = self.get_pixels_in_radius(vector, radius)
        return self.read_halos_in_pixels(pixels, properties)

    def read_halos(self, properties):
        """
        Read all halos in this file and return the requested halo
        properties in a dict of numpy arrays.

        :param properties: list of halo properties (i.e. HDF5 dataset names) to read
        :type  properties: list of str

        :return: dict of arrays with the halo properties
        :rtype:  dict of numpy.ndarray
        """
        return self.read_halos_in_pixels(None, properties)
