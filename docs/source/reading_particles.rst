Reading lightcone particle outputs
==================================

Lightcone particle outputs from SWIFT can be post-processed to allow
faster access to specified areas of the sky and redshift ranges. These
post-processed outputs can be read with the class
:py:class:`lightcone_io.ParticleLightcone`::

  import lightcone_io as lc

  # Specify the name of one of the lightcone particle files
  filename = "./lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"

  # Open the lightcone particle output
  lightcone = lc.ParticleLightcone(filename)

Lightcone particle metadata
---------------------------

The :py:class:`lightcone_io.ParticleLightcone` instance acts like a
dictionary where the particle types are the keys. E.g. to see which
types are available::

  print(list(lightcone))

You can use the properties attribute to see what quantities are available for
each particle type::

  print(lightcone["Gas"].properties)

Each entry in properties is a zero element unyt array with the dtype, units
and shape of the quantity in the file and a ``attrs`` attribute which contains
a copy of the HDF5 attributes of the dataset.

Reading particle data
---------------------

Particles can be read in as follows::

  # Quantities to read in
  properties = ("Coordinates", "ParticleIDs")

  # Position and angular radius (in radians) on the sky to read in
  vector = (1., 0., 0.)
  radius = np.radians(10.)

  # Redshift range to read in
  redshift_range = (0., 1.0)

  # Read dark matter particles
  data = lightcone["DM"].read(property_names, vector, radius, redshift_range)

The return value is a dictionary containing the quantities read in -
in this case Coordinates and ParticleIDs. If ``redshift_range=None``
then all redshifts are read in. If ``vector=None`` and ``radius=None``
then the whole sky is read in.

Note that this may return particles outside the specified region because the
indexed lightcone is stored in chunks and all chunks overlapping the region
are returned. There is also a ``read_exact()`` method which filters out the extra
particles. E.g.::

  data = lightcone["DM"].read_exact(property_names, vector, radius, redshift_range)

This is likely to be slower because it's necessary to read in the coordinates for
spatial selection and the expansion factors for redshift selection even if
these quantities are not being returned. Computing the angles and redshifts
adds some CPU overhead too.

Iterating over chunks of particles
----------------------------------

There is also a way to iterate through the selected particles without reading
them all into memory::

  for data in lightcone["DM"].iterate_chunks(property_names, vector, radius,
                                             redshift_range):
    pos = data["Coordinates"]
    ids = data["ParticleIDs"]
    # then do something with this subset of the selected particles...

Each iteration of this loop will receive a chunk of particles in the dict data.
