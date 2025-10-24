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


def test_read_full_dataset_two_slices(dataset_1d):
    """
    Read all elements in two slices
    """
    nmax = dataset_1d.shape[0]
    starts = (0, nmax//2)
    counts = (nmax//2, nmax-nmax//2)
    result = read_slices(dataset_1d, starts, counts)
    assert result.shape == dataset_1d.shape
    assert np.all(result==dataset_1d[...])
    assert result.dtype == dataset_1d.dtype


def test_read_zero_slices(dataset_1d):
    """
    If we have zero slices, should get a zero sized result.
    """
    nmax = dataset_1d.shape[0]
    starts = ()
    counts = ()
    result = read_slices(dataset_1d, starts, counts)
    assert result.shape == (0,)
    assert result.dtype == dataset_1d.dtype


def test_read_zero_sized_slice(dataset_1d):
    """
    If we have a slice of size zero, should get a zero sized result.
    """
    nmax = dataset_1d.shape[0]
    starts = (0,)
    counts = (0,)
    result = read_slices(dataset_1d, starts, counts)
    assert result.shape == (0,)
    assert result.dtype == dataset_1d.dtype


def test_read_partial_slice(dataset_1d):
    """
    Try reading a single slice smaller than the full dataset
    """
    nmax = dataset_1d.shape[0]
    starts = (100,)
    counts = (200,)
    result = read_slices(dataset_1d, starts, counts)
    assert result.shape == (200,)
    assert result.dtype == dataset_1d.dtype
    assert np.all(dataset_1d[starts[0]:starts[0]+counts[0]] == result)
