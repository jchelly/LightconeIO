#!/bin/env python

import h5py
import healpy as hp
import numpy as np
import unyt
import pytest

from lightcone_io.halo_reader import HaloLightconeFile
from lightcone_io.utils import match

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


def halo_id(pos):
    """
    Generate unique IDs for the halos in the lightcone, based on positions
    """
    nr_halos = pos.shape[0]
    halo_id = np.ndarray(nr_halos, dtype=np.uint64)
    for i in range(nr_halos):
        halo_id[i] = pos[i,:].astype(np.float64).view(np.uint64).sum()
    # As long as these happen to be unique for the test dataset, we're ok!
    assert len(halo_id) == len(np.unique(halo_id))
    return halo_id


def try_read_radius(vector, radius, properties):
    """
    Read halos in the specified radius about a vector and check that we get
    all of the halos within the radius.
    """
    # Read the specified halos and assign unique IDs to them
    partial_halos = HaloLightconeFile(halo_lightcone_filename)
    partial_data = partial_halos.read_halos_in_radius(vector, radius, halo_properties)
    partial_halo_ids = halo_id(partial_data["Lightcone/HaloCentre"])

    # Read all halos and assign unique IDs to them
    with h5py.File(halo_lightcone_filename, "r") as infile:
        full_data = {}
        for name in partial_data:
            full_data[name] = infile[name][...]
        full_halo_ids = halo_id(full_data["Lightcone/HaloCentre"])

    # Identify which halos in the full set were read in
    partial_index = match(full_halo_ids, partial_halo_ids)
    halo_was_read = (partial_index>=0)

    # Check that all arrays agree: if we take the full set of halos and discard
    # any which were not read, we should be left with the partial set.
    assert set(partial_data.keys()) == set(full_data.keys())
    for name in partial_data:
        assert np.all(partial_data[name].value == full_data[name][halo_was_read,...])

    # Then we need to check that all halos in the specified part of the sky
    # were read in. All halos which were not read must be outside the radius.
    for i in range(len(halo_was_read)):
        angle = hp.rotator.angdist(vector, full_data["Lightcone/HaloCentre"][i,:])
        assert (angle > radius) or halo_was_read[i]


def test_read_zero_radius():
    """
    Check what happens if we read no halos
    """
    try_read_radius(np.asarray((1, 0, 0), dtype=float), 0.0, halo_properties)


def random_normalized_vectors(N, rng):
    vectors = rng.normal(size=(N, 3))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    return normalized_vectors


# Generate some randomized test cases
N = 50
rng = np.random.default_rng(seed=0)
test_vectors = random_normalized_vectors(N, rng)
test_radii = rng.random(N) * np.radians(90.0)
test_cases = []
for i in range(N):
    test_cases.append((test_vectors[i,:], test_radii[i]))

@pytest.mark.parametrize("vector,radius", test_cases)
def test_read_radius(vector, radius):
    try_read_radius(vector, radius, halo_properties)

