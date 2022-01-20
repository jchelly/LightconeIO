#!/bin/env python

import h5py
import numpy as np


def find_runs(x):
    """Find runs of consecutive items in an array."""
    x = np.asarray(x)
    if len(x) == 0:
        return np.ndarray(0, dtype=int), np.ndarray(0, dtype=int)
    elif len(x) == 1:
        return np.asarray((0,), dtype=int), np.asarray((1,), dtype=int) 
    else:
        run_start = np.zeros(len(x), dtype=bool)
        run_start[0] = True
        run_start[1:] = x[:-1] != x[1:]-1
        run_start = np.nonzero(run_start)[0]
        run_end   = np.empty_like(run_start)
        run_end[:-1] = run_start[1:]
        run_end[-1] = len(x)
        run_length = run_end - run_start
        return run_start, run_length
    

class HealpixMap(np.ndarray):

    def __new__(cls, input_array, comoving_inner_radius=None,
                comoving_outer_radius=None, nside=None,
                expected_sum=None):
        obj = np.asarray(input_array).view(cls)
        obj.comoving_inner_radius = comoving_inner_radius
        obj.comoving_outer_radius = comoving_outer_radius
        obj.nside = nside
        obj.expected_sum = expected_sum
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.comoving_inner_radius = getattr(obj, 'comoving_inner_radius', None)
        self.comoving_outer_radius = getattr(obj, 'comoving_outer_radius', None)
        self.nside = getattr(obj, 'nside', None)
        self.expected_sum = getattr(obj, 'expected_sum', None)


def read_map(basedir, basename, shell_nr, map_name):
    """
    Read the specified healpix map for a lightcone shell
    """

    # Read the pixel data
    data = []
    file_nr = 0
    nr_files_per_shell = 1
    while file_nr < nr_files_per_shell:
        fname = ("%s/%s_shells/shell_%d/%s.shell_%d.%d.hdf5" % 
                 (basedir, basename, shell_nr, basename, shell_nr, file_nr))
        with h5py.File(fname, "r") as infile:
            data.append(infile[map_name][...])
            if file_nr == 0:
                nr_files_per_shell = infile["Shell"].attrs["nr_files_per_shell"][0]
                nside = infile[map_name].attrs["nside"][0]
                comoving_inner_radius = infile[map_name].attrs["comoving_inner_radius"][0]
                comoving_outer_radius = infile[map_name].attrs["comoving_outer_radius"][0]
                if "expected_sum" in infile[map_name].attrs:
                    expected_sum = infile[map_name].attrs["expected_sum"][0]
                else:
                    expected_sum = None
        file_nr += 1

    data = HealpixMap(np.concatenate(data), nside=nside, 
                      comoving_inner_radius=comoving_inner_radius,
                      comoving_outer_radius=comoving_outer_radius,
                      expected_sum=expected_sum)
    return data


