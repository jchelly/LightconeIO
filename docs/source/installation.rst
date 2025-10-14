Installation
============

Without MPI support
-------------------

If you only want to use this package to read lightcone outputs (such as
the FLAMINGO data release) and don't need MPI support, then
installation is straightforward::

  pip install lightcone_io

With the package installed in this way you can use the
:py:class:`lightcone_io.ShellArray` and
:py:class:`lightcone_io.ParticleLightcone` classes to read lightcone
outputs.

With MPI support
----------------

The :py:class:`lightcone_io.ParticleLightcone` class has the optional
ability to do MPI parallel reading of lightcone particle data. This
requires mpi4py and an MPI enabled build of h5py. There are also
several modules for post-processing SWIFT output which require MPI.

Installing with mpi4py and h5py built from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For MPI support, h5py must be built from source and linked to a
libhdf5 which uses the same MPI installation as mpi4py.

To install mpi4py, ensure that the mpicc from the MPI installation you want
to use is in your $PATH and run::

  pip cache purge
  pip install --no-binary mpi4py mpi4py

To install h5py, assuming that we're using the bash shell::

  export CC="`which mpicc`"
  export HDF5_MPI="ON"
  export HDF5_DIR=<path to mpi enabled hdf5 installation>
  pip cache purge
  pip install setuptools cython numpy pkgconfig
  pip install --no-binary h5py --no-build-isolation h5py

The ``HDF5_DIR`` path should contain HDF5's lib and include directories. You can
then run::

  pip install -e .[mpi]

in the lightcone_io source directory to install the module.

Installing in a virtual environment on Cosma
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If installing on `Cosma <https://cosma.readthedocs.io/en/latest/>`__,
there are prebuilt wheels for mpi4py and h5py which can be used. The
bash script::

  lightcone_io/scripts/virtual_env/make_cosma_env_python3.12.4.sh

will create a new virtual environment in::

  /cosma/apps/dp004/${USER}/lightcone_env

and install the necessary dependencies. If you intend to make any
changes to the lightcone_io module you can then install it in editable
mode by activating the venv and running::

  pip install -e .[mpi]

in the source directory. You should then be able to ``import
lightcone_io`` in python.
