#!/bin/env python

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
        self._filename = filename

        # Open the file and read the index information
        with h5py.File(filename, "r") as infile:
            self._num_halos_per_pixel = infile["Index/NumHalosPerPixel"][...]
            self._nside = int(infile["Index"].attrs["nside"])
            self._order = str(infile["Index"].attrs["order"])

    def read_halos_in_pixels(pixels, properties):
        """
        Read halos in the specified HEALPix pixels and return the
        requested halo properties in a dict of numpy arrays.

        :param pixels: array of HEALPix pixel indexes to read
        :type  pixels: numpy.ndarray
        :param properties: list of halo properties (i.e. HDF5 dataset names) to read
        :type  properties: list of str

        :return: dict of arrays with the halo properties
        :rtype:  dict of numpy.ndarray
        """

        # Determine which parts of the halo arrays we need to read
        
