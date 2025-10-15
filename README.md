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