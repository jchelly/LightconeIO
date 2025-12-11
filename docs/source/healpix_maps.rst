Reading HEALPix maps
====================

When running with lightcones enabled, SWIFT can output HEALPix maps of various
quantities in concentric, spherical shells around the observer. The class
:py:class:`lightcone_io.ShellArray` is used to read these maps. The maps may
be in a single file per shell or split over many files.

The maps can be stored in HDF5 files on the local file system or in
remote files accessed via a web service.

Opening local HDF5 files
------------------------

If the outputs you wish to read are stored as HDF5 files on your local
filesystem, you can open them as follows::

  import lightcone_io as lc

  # Location of the lightcone output
  basedir="./lightcones/"

  # Which lightcone to read
  basename="lightcone0"

  # Open a set of HEALPix maps
  shell = lc.ShellArray(basedir, basename)

You can then access the maps through the ``shell`` object. See below
for details.

Opening via the hdfstream service
---------------------------------

This module can also access files stored on a remote server using the
`hdfstream <https://hdfstream-python.readthedocs.io/en/latest>`_
python module. To do this, first open the directory containing the
files::

  import hdfstream
  root = hdfstream.open("cosma", "/")

This returns a :obj:`hdfstream.RemoteDirectory` object. HEALPix maps
on the server can be read by passing the remote directory object to
the :py:class:`lightcone_io.ShellArray` class. The base directory name
is interpreted as a path relative to the remote directory on the
server::

  import lightcone_io as lc

  # Location of the lightcone output relative to the remote directory
  basedir="FLAMINGO/L1_m9/L1_m9/healpix_maps/nside_4096"

  # Which lightcone to read
  basename="lightcone0"

  # Open a set of HEALPix maps on the server
  shell = lc.ShellArray(basedir, basename, remote_dir=root)

You can then access the maps through the ``shell`` object. The module
will download data as it is accessed.

Accessing HEALPix map data
--------------------------

The :py:class:`lightcone_io.ShellArray` object is a sequence of
:py:class:`lightcone_io.Shell` instances. Individual shells are
accessed by indexing with the shell number::

  # Print the number of shells
  print(len(shell))

  # Print the inner and outer raddi of a single shell
  print(shell[0].comoving_inner_radius, shell[0].comoving_outer_radius)

Each shell contains one or more :py:class:`lightcone_io.HealpixMap`
instances, which are spherical maps of physical quantities. These maps
are accessed with dictionary style indexing::

  # Print names of the available maps for the first shell
  print(list(shell[0]))

  # Find the TotalMass map for the first shell
  total_mass_map = shell[0]["TotalMass"]

  # Return some information about this map
  print(total_mass_map.nside) # HEALPix nside parameter
  print(total_mass_map.dtype) # Data type of the pixel data
  print(total_mass_map.units) # Units of the pixel data
  print(len(total_mass_map))  # Total number of pixels

The pixel data can be read in by indexing the map. Simple ``[start:end]`` slices
can be used to read subsets of the pixels or Ellipses (``...``) can be used to
read all of the pixels::

  # Read all of the pixels from the TotalMass map of the first shell
  pixel_data = shell[0]["TotalMass"][...]

  # Or read just the first 100 pixels
  pixel_data_partial = shell[0]["TotalMass"][0:100]

The results are returned as a unyt array with unit information derived
from the HDF5 attributes in the output files.
