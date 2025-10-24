#!/bin/env python

import io
import h5py
import numpy as np
import pytest

from lightcone_io.utils import read_slices


@pytest.fixture(scope='module')
def dataset_1d():
    """
    Make an in-memory dataset for testing
    """
    nmax = 10000
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        data = np.arange(nmax, dtype=int)
        f["data"] = data
        yield f["data"]


def test_read_full_dataset(dataset_1d):
    """
    Test the trivial case of reading all elements in one slice
    """
    nmax = dataset_1d.shape[0]
    starts = (0,)
    counts = (nmax,)
    result = read_slices(dataset_1d, starts, counts)
    assert result.shape == dataset_1d.shape
    assert np.all(result==dataset_1d[...])
    assert result.dtype == dataset_1d.dtype

