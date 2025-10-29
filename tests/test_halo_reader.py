#!/bin/env python

import h5py
import numpy as np
import unyt
import pytest

from lightcone_io.halo_reader import HaloLightconeFile

halo_lightcone_filename = "./tests/data/halo_lightcone/lightcone_halos_0070.hdf5"
soap_filename = "./tests/data/halo_lightcone/halo_properties_0070.hdf5"

# Quantities which should exist in the halo lightcone file
halo_properties = (
    "InputHalos/HaloCatalogueIndex",
    "InputHalos/HaloCentre",
    "InputHalos/SOAPIndex",
    "InputHalos/HBTplus/TrackId",
    "Lightcone/HaloCentre",
    "Lightcone/Redshift",
    "Lightcone/SnapshotNumber",
)

# Extra quantities we can get from SOAP
soap_properties = (
    "SO/200_crit/TotalMass",
)


def test_properties():
    """
    Check that the file has the expected set of properties
    """
    halos = HaloLightconeFile(halo_lightcone_filename)
    assert set(halo_properties) == set(halos.properties)


def test_read_everything():
    """
    Try reading all properties for all halos
    """
    # Read the file
    data = HaloLightconeFile(halo_lightcone_filename).read_halos(halo_properties)

    # Check values against h5py
    with h5py.File(halo_lightcone_filename, "r") as infile:
        for name in halo_properties:
            assert isinstance(data[name], unyt.unyt_array)
            assert np.all(data[name].value == infile[name][...])


def test_read_everything_with_soap():
    """
    Try reading all properties for all halos. This version reads m200c from
    the SOAP file too.
    """
    # Open the file
    halos = HaloLightconeFile(halo_lightcone_filename, soap_filename=soap_filename)

    # Read all properties
    data = halos.read_halos(halo_properties+soap_properties)

    # Check values against halo lightcone file
    with h5py.File(halo_lightcone_filename, "r") as infile:
        for name in halo_properties:
            assert isinstance(data[name], unyt.unyt_array)
            assert np.all(data[name].value == infile[name][...])

    # Check values against SOAP
    soap_index = data["InputHalos/SOAPIndex"].value
    with h5py.File(soap_filename, "r") as infile:
        for name in soap_properties:
            assert isinstance(data[name], unyt.unyt_array)
            assert np.all(data[name].value == infile[name][...][soap_index,...])
