#!/bin/env python

import io
import h5py
import numpy as np
import pytest

from lightcone_io.utils import read_slices


@pytest.fixture(scope='module', params=[(), (0,), (1,), (3,), (3,2)])
def dataset(request):
    """
    Make an in-memory dataset for testing.

    Here we test slicing along the first dimension. The fixture param here
    contains the dimensions of the dataset in the remaining dimensions.
    """
    # Size in the first dimension
    nr_first_dim = 10000

    # Full shape of the test dataset
    shape = (nr_first_dim,)+request.param

    # Compute total number of elements
    nr_total = nr_first_dim
    for n in shape[1:]:
        nr_total *= n
    # Make an array with the test data
    data = np.arange(nr_total, dtype=int).reshape(shape)

    # Write the HDF5 dataset
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        f["data"] = data

    # Reopen in read only mode
    bio.seek(0)
    with h5py.File(bio, 'r') as f:
        yield f["data"]


def test_read_full_dataset(dataset):
    """
    Test the trivial case of reading all elements in one slice
    """
    nmax = dataset.shape[0]
    starts = (0,)
    counts = (nmax,)
    result = read_slices(dataset, starts, counts)
    assert result.shape == dataset.shape
    assert np.all(result==dataset[...])
    assert result.dtype == dataset.dtype


def test_read_full_dataset_two_slices(dataset):
    """
    Read all elements in two slices
    """
    nmax = dataset.shape[0]
    starts = (0, nmax//2)
    counts = (nmax//2, nmax-nmax//2)
    result = read_slices(dataset, starts, counts)
    assert result.shape == dataset.shape
    assert np.all(result==dataset[...])
    assert result.dtype == dataset.dtype


def test_read_zero_slices(dataset):
    """
    If we have zero slices, should get a zero sized result.
    """
    nmax = dataset.shape[0]
    starts = ()
    counts = ()
    result = read_slices(dataset, starts, counts)
    assert result.shape[0] == 0
    assert result.shape[1:] == dataset.shape[1:]
    assert result.dtype == dataset.dtype


def test_read_zero_sized_slice(dataset):
    """
    If we have a slice of size zero, should get a zero sized result.
    """
    nmax = dataset.shape[0]
    starts = (0,)
    counts = (0,)
    result = read_slices(dataset, starts, counts)
    assert result.shape[0] == 0
    assert result.shape[1:] == dataset.shape[1:]
    assert result.dtype == dataset.dtype


def test_read_partial_slice(dataset):
    """
    Try reading a single slice smaller than the full dataset
    """
    nmax = dataset.shape[0]
    starts = (100,)
    counts = (200,)
    result = read_slices(dataset, starts, counts)
    assert result.shape[0] == 200
    assert result.shape[1:] == dataset.shape[1:]
    assert result.dtype == dataset.dtype
    assert np.all(dataset[starts[0]:starts[0]+counts[0],...] == result)
