#!/bin/env python

import h5py
import healpy as hp
import numpy as np
import unyt
import pytest

from lightcone_io import ShellArray, Shell, HealpixMap

basedir  = "./tests/data/healpix_maps"
basename = "lightcone1"
shell_filename = "./tests/data/healpix_maps/lightcone1_shells/shell_{shell_nr}/lightcone1.shell_{shell_nr}.0.hdf5"
maps = ("DarkMatterMass", "NeutrinoMass", "TotalMass")
nside = 16
nr_pixels = hp.nside2npix(nside)
nr_shells = 10


def test_open_shellarray():
    """
    Check we can open an array of healpix map shells
    """
    shells = ShellArray(basedir, basename)
    assert len(shells) == nr_shells
    for shell in shells:
        assert isinstance(shell, Shell)


def test_shell_info():
    """
    Check that shell metadata is correct
    """
    shells = ShellArray(basedir, basename)
    for shell_nr, shell in enumerate(shells):
        assert set(maps) == set(shell.keys())
        with h5py.File(shell_filename.format(shell_nr=shell_nr), "r") as infile:
            r_inner = infile["Shell"].attrs["comoving_inner_radius"][0]
            r_outer = infile["Shell"].attrs["comoving_outer_radius"][0]
        assert isinstance(shell.comoving_inner_radius, unyt.unyt_quantity)
        assert isinstance(shell.comoving_outer_radius, unyt.unyt_quantity)
        assert shell.comoving_inner_radius.value == r_inner
        assert shell.comoving_outer_radius.value == r_outer


def test_map_info():
    """
    Check that healpix map metadata is correct
    """
    shells = ShellArray(basedir, basename)
    shell_nr = 0
    shell = shells[shell_nr]
    with h5py.File(shell_filename.format(shell_nr=shell_nr), "r") as infile:
        for name in shell:
            map_data = shell[name]
            assert map_data.nside == nside
            cgs_conversion = infile[name].attrs["Conversion factor to CGS (not including cosmological corrections)"][0]
            assert np.isclose(cgs_conversion, shell[name].units.in_cgs().value)
            assert map_data.dtype == infile[name].dtype
            assert len(map_data) == infile[name].attrs["number_of_pixels"][0]


def test_full_map():
    """
    Read full maps and check pixel values against h5py
    """
    shells = ShellArray(basedir, basename)
    for shell_nr, shell in enumerate(shells):
        with h5py.File(shell_filename.format(shell_nr=shell_nr), "r") as infile:
            for name in shell:
                assert np.all(shell[name][...].value == infile[name][...])


# Generate some random test cases for reading pixel subsets
N = 50
rng = np.random.default_rng(seed=0)
map_index = rng.integers(0, len(maps), N)
shell_nr = rng.integers(0, nr_shells, N)
offset = rng.integers(0, nr_pixels, N)
length = rng.integers(0, nr_pixels, N)
test_cases = []
for i in range(N):
    test_cases.append((shell_nr[i], maps[map_index[i]], offset[i], min(length[i], nr_pixels-offset[i])))


@pytest.mark.parametrize("shell_nr,name,offset,length", test_cases)
def test_read_pixels(shell_nr, name, offset, length):
    """
    Try reading some pixels from a map and check against h5py
    """
    map_data = ShellArray(basedir, basename)[shell_nr][name]
    pixels = map_data[offset:offset+length].value
    with h5py.File(shell_filename.format(shell_nr=shell_nr), "r") as infile:
        assert np.all(pixels == infile[name][offset:offset+length])
