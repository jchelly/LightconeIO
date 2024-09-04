#!/bin/bash
#
# Set up a virtual env suitable for running mpi4py code using openmpi and
# parallel HDF5.
#

set -e

ompi_version=5.0.3
hdf5_version=1.12.3

module purge
module load python/3.12.4

# Pip can't distinguish between incompatible builds of the same module,
# so we need to clear any cached wheels before we start.
python -m pip cache purge

# Location of local wheels
WHEEL_DIR=/cosma/local/python-wheels/3.12.4/openmpi-${ompi_version}-hdf5-${hdf5_version}

# Name of the new venv to create
venv_name="/cosma/apps/dp004/${USER}/lightcone_env"

# Create an empty venv
python -m venv "${venv_name}"

# Activate the venv
source "${venv_name}"/bin/activate

# Install modules with tricky dependencies from locally built wheels
pip install ${WHEEL_DIR}/mpi4py-3.1.6-cp312-cp312-linux_x86_64.whl
pip install ${WHEEL_DIR}/h5py-3.11.0-cp312-cp312-linux_x86_64.whl

# Install lightcone I/O to the venv in editable mode
cd ../..
pip3 install -e .
