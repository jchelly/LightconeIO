#!/bin/env python

import io
import h5py
import numpy as np
import pytest

from lightcone_io.utils import read_slices, SlicedDatasetReader


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


# Test cases
nmax = 10000
test_cases = [
    ((0,),         (nmax,)),                   # All in one slice
    ((0, nmax//2), (nmax//2, nmax-nmax//2)),   # All in two slices
    ((),           ()),                        # Zero slices
    ((0,),         (0,)),                      # A zero sized slice
    ((100,),       (200,)),                    # A single partial slice
]


@pytest.mark.parametrize("starts,counts", test_cases)
def test_read_slice(dataset, starts, counts):
    """
    Check that read_slices() reads slices correctly
    """
    # Use read_slices() to read the data
    data1 = read_slices(dataset, starts, counts)

    # Try reading the slices directly with h5py
    data2 = []
    for s, c in zip(starts, counts):
        data2.append(dataset[s:s+c,...])
    if len(data2) > 0:
        data2 = np.concatenate(data2, axis=0)
    else:
        shape = (0,)+dataset.shape[1:]
        data2 = np.zeros(shape, dtype=dataset.dtype)

    # Check the result
    assert data1.shape == data2.shape
    assert data1.dtype == data2.dtype
    assert np.all(data1==data2)


@pytest.mark.parametrize("starts,counts", test_cases)
def test_sliced_dataset_reader(dataset, starts, counts):
    """
    Check that SlicedDatasetReader reads slices correctly
    """
    # Use SlicedDatasetReader to read the data
    data1 = SlicedDatasetReader(starts, counts).read(dataset)

    # Try reading the slices directly with h5py
    data2 = []
    for s, c in zip(starts, counts):
        data2.append(dataset[s:s+c,...])
    if len(data2) > 0:
        data2 = np.concatenate(data2, axis=0)
    else:
        shape = (0,)+dataset.shape[1:]
        data2 = np.zeros(shape, dtype=dataset.dtype)

    # Check the result
    assert data1.shape == data2.shape
    assert data1.dtype == data2.dtype
    assert np.all(data1==data2)
