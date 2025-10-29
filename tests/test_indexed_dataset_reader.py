#!/bin/env python

import h5py
import numpy as np
import pytest

from lightcone_io.utils import read_slices, IndexedDatasetReader


# Some simple test cases
test_cases = [
    (),               # Empty index array
    (0,),             # Just one index at the start
    (5000,),          # Just one index somewhere else
    np.arange(10000, dtype=int),       # All elements
    np.arange(1000, dtype=int) + 4000, # A contiguous chunk
    np.arange(1000, dtype=int) * 2,    # Even elements
    np.arange(1000, dtype=int) * 2 + 1,# Odd elements
]

def random_elements(rng, nmax):
    """
    Make a random array of elements to read
    """
    # Decide how many elements to read
    n = rng.integers(0, nmax, 1)[0]
    # Pick the indexes to read
    return rng.integers(0, nmax, size=n)

# Add some random test cases with non-unique, non-sorted indexes
rng = np.random.default_rng(seed=0)
for _ in range(10):
    test_cases.append(random_elements(rng, 10000))


@pytest.mark.parametrize("index", test_cases)
def test_read_slice(dataset, index):
    """
    Check that IndexedDatasetReader reads datasets correctly
    """

    # Read the data using the IndexedDatasetReader
    data1 = IndexedDatasetReader(index).read(dataset)

    # Read all of the data with h5py
    if len(index) > 0:
        # Read everything and have numpy extract the elements we want
        data2 = dataset[...][index,...]
    else:
        # Handle the case of reading zero elements
        shape = (0,)+dataset.shape[1:]
        data2 = np.zeros(shape, dtype=dataset.dtype)

    # Check the result
    assert data1.shape == data2.shape
    assert data1.dtype == data2.dtype
    assert np.all(data1==data2)
