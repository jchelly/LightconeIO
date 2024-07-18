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
WHEEL_DIR=/cosma/apps/dp004/jch/python/wheels/python3.12.4/openmpi-${ompi_version}-hdf5-${hdf5_version}

# Name of the new venv to create
venv_name="lightcone_env"

# Create an empty venv
python -m venv "${venv_name}"

# Activate the venv
source "${venv_name}"/bin/activate

# Install modules with tricky dependencies from locally built wheels
pip install ${WHEEL_DIR}/mpi4py-3.1.6-cp312-cp312-linux_x86_64.whl
pip install ${WHEEL_DIR}/h5py-3.11.0-cp312-cp312-linux_x86_64.whl
