#!/bin/env python

import h5py
import hdfstream
import contextlib


class LocalOrRemoteFile:
    """
    Mixin class used to open local or remote files
    """
    def set_directory(self, remote_dir=None):
        self._remote_dir = remote_dir

    @contextlib.contextmanager
    def open_file(self, filename):
        if getattr(self, "_remote_dir", None) is None:
            # We're opening a local file with h5py
            with h5py.File(filename, "r") as f:
                yield f
        else:
            # We're opening a remote file with hdfstream
            yield self._remote_dir[filename]
