#!/bin/env python

import numpy as np
import h5py
import healpy as hp
import unyt

from .units import units_from_attributes
from .utils import IndexedDatasetReader, SlicedDatasetReader


class HaloLightconeFile:
    """
    Class used to read a single file from the halo lightcone. Each
    file contains halos which were identified in a single snapshot of
    the simulation. Halos are sorted according to their pixel index in
    a low resolution HEALPix map so that regions on the sky can be
    extracted efficiently.

    If the name of the SOAP file which was used to generate the lightcone
    halo catalogue is supplied, then it will be possible to look up extra
    halo properties from SOAP.

    :param filename: Name of the halo lightcone file to open
    :type  filename: str
    :param soap_filename: Name of the SOAP output with extra halo properties
    :type  soap_filename: str
    """
    def __init__(self, filename, soap_filename=None):
        self._file = h5py.File(filename, "r")
        if soap_filename is not None:
            self._soap_file = h5py.File(soap_filename, "r")
        else:
            self._soap_file = None
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

        Note that this does not include any additional properties which we
        may be able to read from SOAP.
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
        requested halo properties in a dict of unyt arrays.

        If a SOAP filename was specifed then we'll check the SOAP file for any
        halo properties which are not present in the halo lightcone file.

        :param pixels: array of HEALPix pixel indexes to read, or None to read all
        :type  pixels: numpy.ndarray, or None
        :param properties: list of halo properties (i.e. HDF5 dataset names) to read
        :type  properties: list of str

        :return: dict of arrays with the halo properties
        :rtype:  dict of unyt.unyt_array
        """

        # Name of the dataset in the halo lightcone which specifies the
        # corresponding index in SOAP.
        soap_index_name = "InputHalos/SOAPIndex"

        # Determine which, if any, properties are to be read from SOAP and
        # which are to be read from the lighcone.
        lightcone_properties = []
        soap_properties = []
        for name in properties:
            if name in self._file:
                lightcone_properies.append(name)
            elif self._soap_file is not None and name in self._soap_file:
                soap_properies.append(name)
            else:
                raise KeyError(f"Unable to locate halo property: {name}")

        # If we're reading anything from SOAP we'll need the SOAP index of each
        # halo, so make sure it's in the list of things to read
        if len(soap_properties) > 0 and soap_index_name not in lightcone_properties:
            lightcone_properies.append(soap_index_name)

        if pixels is not None:
            # Determine ranges of halos to read
            if np.any(pixels[1:] <= pixels[:-1]):
                raise RuntimeError("Pixel indexes must be unique and ascending!")

            # Discard any selected pixels with no halos
            pixels = pixels[self._num_halos_per_pixel[pixels] > 0]

            # Compute offset to first halo and number of halos per pixel
            offsets = self._first_halo_in_pixel[pixels]
            counts  = self._num_halos_per_pixel[pixels]

        else:
            # We're reading all of the halos, so we just have one range to read
            offsets = np.asarray((0,), dtype=int)
            counts = np.asarray((np.sum(self._num_halos_per_pixel),), dtype=int)

        # Compute expected number of halos
        nr_halos = sum(counts)

        # Read halo properties from the lightcone
        result = {}
        reader = SlicedDatasetReader(offsets, counts)
        for name in lightcone_properties:
            dataset = self._file[name]
            units = units_from_attributes(dataset)
            result[name] = unyt.unyt_array(reader.read(dataset), units)

        # Now read extra properties from SOAP, if necessary
        if len(soap_properties) > 0:

            # Get the index of each selected lightcone halo
            soap_index = data[soap_index_name].value

            # Read these indexes from the SOAP datasets
            reader = IndexedDatasetReader(soap_index)
            for name in soap_properties:
                dataset = self._soap_file[name]
                units = units_from_attributes(dataset)
                result[name] = unyt.unyt_array(reader.read(dataset), units)

        # If the SOAP index was not requested, don't return it
        if soap_index_name in result and soap_index_name not in properties:
            del result[soap_index_name]

        return result

    def read_halos_in_radius(self, vector, radius, properties):
        """
        Read halos in an angular radius around the specified line of sight
        vector and return the requested halo properties in a dict of unyt
        arrays. May also return some halos slightly outside the radius
        because if we read a pixel we read all of the halos in it.

        :param vector: direction vector as an array of 3 floats
        :type  vector: numpy.ndarray
        :param radius: angular radius in radians
        :type  radius: float
        :param properties: list of halo properties (i.e. HDF5 dataset names) to read
        :type  properties: list of str

        :return: dict of arrays with the halo properties
        :rtype:  dict of unyt.unyt_array
        """
        pixels = self.get_pixels_in_radius(vector, radius)
        return self.read_halos_in_pixels(pixels, properties)

    def read_halos(self, properties):
        """
        Read all halos in this file and return the requested halo
        properties in a dict of unyt arrays.

        :param properties: list of halo properties (i.e. HDF5 dataset names) to read
        :type  properties: list of str

        :return: dict of arrays with the halo properties
        :rtype:  dict of unyt.unyt_array
        """
        return self.read_halos_in_pixels(None, properties)
