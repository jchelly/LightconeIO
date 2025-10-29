#!/bin/env python

import io
import h5py
import numpy as np
import pytest


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
