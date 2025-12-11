# Lightcone I/O for SWIFT

This is a python module for reading lightcone output from SWIFT simulations.

The source code and issue tracker are hosted on github:
https://github.com/jchelly/LightconeIO

Releases are hosted on pypi: https://pypi.org/project/lightcone-io/

For documentation see: https://lightconeio.readthedocs.io/en/latest/

## Installation

For read-only use without MPI support, the module can be installed using pip:
```
pip install lightcone_io
```

## Building the documentation

The documentation is built using sphinx:
```
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
cd docs
make html
```

## Running unit tests

There are some unit tests which can be run with just
```
pytest
```
in the source directory.  This will test the classes for reading local
HDF5 files containing HEALPix map, particle and halo lightcone
data. These tests use downsampled lightcone outputs which are stored
in the git repository in `tests/data`.

To also test access to remote files:
```
pytest --hdfstream-server=<server_url>
```
