#!/bin/env python

import io
import h5py
import numpy as np
import pytest

from lightcone_io.utils import merge_slices, validate_slices


def count_times_selected(n, starts, counts):
    """
    Count how many times each of n array elements is in a slice
    """
    nr_selections = np.zeros(n, dtype=int)
    for s, c in zip(starts, counts):
        nr_selections[s:s+c] += 1

    # Elements should all be selected 0-1 times
    assert np.all(nr_selections>=0)
    assert np.all(nr_selections<=1)

    return nr_selections


def assert_slices_equivalent(slicespec1, slicespec2):
    """
    Check that slicespec1 and slicespec2 specify the same elements

    Each slicespec is a (starts, counts) tuple where starts is an array
    with the offset to each slice and counts is an array with the length
    of each slice.
    """
    starts1, counts1 = slicespec1
    validate_slices(starts1, counts1)
    starts2, counts2 = slicespec2
    validate_slices(starts2, counts2)

    # Determine the maximum array index
    if len(starts1) > 0:
        n1 = np.amax(starts1+counts1)
    else:
        n1 = 0
    if len(starts2) > 0:
        n2 = np.amax(starts2+counts2)
    else:
        n2 = 0
    n = max(n1, n2)

    # Check that the slices select the same elements
    selections1 = count_times_selected(n, starts1, counts1)
    selections2 = count_times_selected(n, starts2, counts2)
    assert np.all(selections1==selections2)


def random_slices(nr_slices, max_length, rng):
    """
    Generate some random test cases for test_merge_slices()
    """
    # Maximum length of a slice or gap between slices
    max_length = 20

    # Draw lengths of slices and gaps between them
    lengths = rng.integers(0, max_length+1, size=2*nr_slices)

    # Generate the slices
    all_starts = []
    all_counts = []
    offset = 0
    for i in range(nr_slices):
        # Skip some elements before this slice
        offset += lengths[2*i+0]
        # Define this slice
        start = offset
        count = lengths[2*i+1]
        # Advance past this slice
        offset += count
        # Store the slice
        all_starts.append(start)
        all_counts.append(count)
    return np.asarray(all_starts, dtype=int), np.asarray(all_counts, dtype=int)


# Some simple test cases for test_merge_slices()
test_slices = [
    ([],        []),        # an empty list of slices
    ([0,],      [0,]),      # one zero length slice
    ([0, 1],    [0, 0]),    # two zero length slices
    ([0,],      [10,]),     # a single slice
    ([5,],      [10,]),     # a single slice with non-zero start
    ([0, 5],    [5, 5]),    # two slices which can be merged
    ([0, 5, 5], [5, 0, 5]), # two slices which can be merged, separated by a zero size slice
    ([0, 6],    [5, 5]),    # two slices which cannot be merged
]

# Add some random test cases
rng = np.random.default_rng(seed=0)
for _ in range(50):
    nr_slices, max_length = rng.integers(0, 10, size=2)
    test_slices.append(random_slices(nr_slices, max_length, rng))


@pytest.mark.parametrize("starts,counts", test_slices)
def test_merge_slices(starts, counts):
    """
    Check that merging slices does not change the selected array indexes
    """
    starts = np.asarray(starts, dtype=int)
    counts = np.asarray(counts, dtype=int)
    merged_starts, merged_counts = merge_slices(starts, counts)
    assert_slices_equivalent((starts, counts), (merged_starts, merged_counts))
