#!/bin/env python

import h5py
import healpy as hp
import numpy as np
import unyt
import pytest

from lightcone_io import ParticleLightcone
from lightcone_io.utils import match


# Names of the particle files
particle_filename = "./tests/data/particles/lightcone0_particles/lightcone0_0000.{file_nr}.hdf5"
nr_particle_files = 4

# Expected contents of the files
particle_types = ("BH",)
particle_properties = {
    "Coordinates"      : ((0,3), np.float64, 3.08567758e+24),
    "DynamicalMasses"  : ((0,),  np.float32, 1.98841e+43),
    "ExpansionFactors" : ((0,),  np.float32, None),
    "ParticleIDs"      : ((0,),  np.int64,   None),
}

# Range of redshifts covered by the particle lightcone
particle_z_min = 0.0
particle_z_max = 15.0


def test_lightcone_metadata():
    """
    Check that we're reading metadata correctly
    """
    lightcone = ParticleLightcone(particle_filename.format(file_nr=0))
    assert set(lightcone) == set(particle_types)

    # Check we have the expected properties
    for ptype in particle_types:
        for name in lightcone[ptype].properties:
            shape, dtype, units_cgs = particle_properties[name]
            assert lightcone[ptype].properties[name].shape == shape
            assert lightcone[ptype].properties[name].dtype == dtype
            if units_cgs is not None:
                assert np.isclose(units_cgs, (1.0*lightcone[ptype].properties[name].units).in_cgs().value)
            else:
                assert lightcone[ptype].properties[name].units == unyt.dimensionless


def test_read_all():
    """
    Try reading in a full array and comparing to result from h5py
    """
    # Read using the ParticleLightcone class
    lightcone = ParticleLightcone(particle_filename.format(file_nr=0))
    pos1 = lightcone["BH"].read(("Coordinates",))["Coordinates"]
    assert isinstance(pos1, unyt.unyt_array)

    # Read using h5py
    pos2 = []
    for file_nr in range(nr_particle_files):
        with h5py.File(particle_filename.format(file_nr=file_nr), "r") as infile:
            pos2.append(infile["BH/Coordinates"][...])
    pos2 = np.concatenate(pos2)

    # Compare values
    assert np.all(pos1.value==pos2)


def particle_id(pos):
    """
    Generate unique IDs for the particles in the lightcone, based on positions
    """
    nr_particles = pos.shape[0]
    particle_id = np.ndarray(nr_particles, dtype=np.uint64)
    for i in range(nr_particles):
        particle_id[i] = pos[i,:].astype(np.float64).view(np.uint64).sum()
    # As long as these happen to be unique for the test dataset, we're ok!
    assert len(particle_id) == len(np.unique(particle_id))
    return particle_id


test_cases = [
    (None,            None,             None),       # read everything
    (None,            None,             (5.0, 6.0)), # redshift selection only
    ((1.0, 0.0, 0.0), np.radians(10.0), None),       # radius selection only
    ((1.0, 0.0, 0.0), np.radians(10.0), (5.0, 6.0)), # select on redshift and radius
]
@pytest.mark.parametrize("vector,radius,redshift_range", test_cases)
def test_redshift_range(vector, radius, redshift_range):
    """
    Try reading a redshift range and patch on the sky and comparing to h5py
    """

    # Read the data using the ParticleLightcone class
    lightcone = ParticleLightcone(particle_filename.format(file_nr=0))
    partial_data = lightcone["BH"].read(("Coordinates", "ParticleIDs", "ExpansionFactors"), vector, radius, redshift_range)

    # Read all of the data using h5py
    full_data = {"Coordinates" : [], "ParticleIDs" : [], "ExpansionFactors" : []}
    for file_nr in range(nr_particle_files):
        with h5py.File(particle_filename.format(file_nr=file_nr), "r") as infile:
            for name in full_data:
                full_data[name].append(infile["BH"][name][...])
    for name in full_data:
        full_data[name] = np.concatenate(full_data[name])

    # Assign unique particle IDs
    partial_data["UniqueID"] = particle_id(partial_data["Coordinates"])
    full_data["UniqueID"] = particle_id(full_data["Coordinates"])

    # Identify which particles in the full set were read in
    partial_index = match(full_data["UniqueID"], partial_data["UniqueID"])
    particle_was_read = (partial_index>=0)

    # Check that all arrays agree: if we take the full set of halos and discard
    # any which were not read, we should be left with the partial set.
    assert set(partial_data.keys()) == set(full_data.keys())
    for name in partial_data:
        if name != "UniqueID":
            assert np.all(partial_data[name].value == full_data[name][particle_was_read,...])

    # Identify particles in the requested redshift range
    if redshift_range is not None:
        full_z = 1.0/full_data["ExpansionFactors"] - 1.0
        in_z_range = (full_z >= min(redshift_range)) & (full_z <= max(redshift_range))
    else:
        in_z_range = np.ones_like(particle_was_read)

    # Identify particles in the requested patch of sky
    in_radius = np.ones_like(particle_was_read)
    if vector is not None:
        for i in range(len(particle_was_read)):
            angle = hp.rotator.angdist(vector, full_data["Coordinates"][i,:])
            in_radius[i] = (angle <= radius)

    # All particles which were not read should be outside the selection
    selected = in_z_range & in_radius
    assert np.all(particle_was_read | np.logical_not(selected))
