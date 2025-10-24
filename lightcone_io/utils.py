#!/bin/env python

import h5py
import numpy as np

def read_slices(dataset, starts, counts, result=None):
    """
    Read the specified slices from a HDF5 dataset. Uses h5py low level calls
    to read the slices with a single H5Dread(). Datasets can only be sliced
    along the first dimension: we always read all elements in the remaining
    dimensions.

    Slices must be in ascending order of starting index and must not overlap.
    Python/numpy style negative indexes from the end of the dataset are not
    supported.

    :param dataset: HDF5 dataset to read from
    :type  dataset: h5py.Dataset
    :param starts: 1D array with starting offset of each slice
    :type  starts: np.ndarray
    :param counts: 1D array with length of each slice
    :type  counts: np.ndarray
    :param result: array to hold the result
    :type  result: np.ndarray, or None

    :return: a numpy array with the data
    :rtype: numpy.ndarray
    """

    # Sanity check the slices
    starts = np.asarray(starts, dtype=int)
    counts = np.asarray(counts, dtype=int)
    if starts.shape != counts.shape:
        raise RuntimeError("start and count arrays must be the same shape")
    if len(starts.shape) != 1 or len(counts.shape) != 1:
        raise RuntimeError("start and count arrays must be 1D")
    if len(starts) > 1:
        if np.any(starts[1:] < starts[:-1]):
            raise RuntimeError("slices must be in ascending order of start index")
        ends = starts + counts
        if np.any(starts[1:] < ends[:-1]):
            raise RuntimeError("slices must not overlap")
    if np.any(counts < 0):
        raise RuntimeError("slices must have non-negative counts")
    if np.any(starts < 0):
        # We don't support negative indexes
        raise RuntimeError("slices must have non-negative starts")

    # Get dataset handle
    dataset_id = dataset.id

    # Get file dataspace handle
    file_space_id = dataset_id.get_space()
    file_shape = file_space_id.get_simple_extent_dims()

    # Select the slices to read
    nr_in_first_dim = 0
    file_space_id.select_none()
    for start, count in zip(starts, counts):
        if count > 0:
            # Select this slice
            slice_start = tuple([start,]+[0 for fs in file_shape[1:]])
            slice_count = tuple([count,]+[i for fs in file_shape[1:]])
            file_space_id.select_hyperslab(slice_start, slice_count, op=h5py.h5s.SELECT_OR)
            nr_in_first_dim += count

    # Allocate the output array, if necessary
    result_shape = [nr_in_first_dim,]+list(file_shape[1:])
    result_shape = tuple([int(rs) for rs in result_shape])
    if result is None:
        result = np.ndarray(result_shape, dtype=dataset.dtype)

    # Output array must be C contiguous
    if not result.flags['C_CONTIGUOUS']:
        raise RuntimeError("Can only read into C contiguous arrays!")

    # Output array must have the expected number of elements
    nr_selected = file_space_id.get_select_npoints()
    if nr_selected != result.size:
        raise RuntimeError("Output buffer is not the right size for the selected slices!")

    # The output array must have the expected shape (could be wrong if it was passed in)
    if result.shape != result_shape:
        raise RuntimeError("Output buffer has the wrong shape!")

    # If we selected any elements, read the data
    if nr_in_first_dim > 0:
        mem_space_id = h5py.h5s.create_simple(result_shape)
        dataset_id.read(mem_space_id, file_space_id, result)

    return result
