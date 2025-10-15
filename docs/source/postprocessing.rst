Post-processing SWIFT outputs
=============================

This section explains how to post-process raw lightcone output from
SWIFT into a more usable form.

.. note::
   You can ignore this section if you're using ``lightcone_io`` to access
   the FLAMINGO data release, which has already been post processed.

Combining HEALPix maps
----------------------

This package can read SWIFT HEALPix output regardless of how many files it
is split over. However, it may be desirable to reduce the number of files if
they're stored on a file system optimized for small numbers of large files.

The module ``lightcone_io.combine_maps`` can be used to combine the
maps for each shell into a single HDF5 file. This code is parallelized
using mpi4py and can be run as follows::

  input_dir=./lightcones/
  output_dir=./indexed_lightcones/

  mpirun python3 -m mpi4py -m lightcone_io.combine_maps \
         ${input_dir} ${output_dir} lightcone0 lightcone1 ...

This will process all shells for the specified lightcones. There is an
example SLURM batch script to run on the FLAMINGO simulations on
COSMA-8 in scripts/FLAMINGO/combine_L1000N1800.sh.

Indexing particle outputs
-------------------------

SWIFT lightcone particle outputs are spread over many files and not sorted
in any useful order. The module ``lightcone_io.index_particles`` can be
used to sort the particles and generate an index which can be used to
quickly retrieve particles by redshift and position on the sky.

The sky is divided into pixels using a low resolution HEALPix map and
each pixel is split into redshift bins. This defines a set of cells of
varying volume. The redshift bins are chosen such that the number of
particles per cell is roughly constant. The particles are then stored
in order of which cell they belong to and the location of each cell in
the output files is stored. This information is used by the
:py:class:`lightcone_io.ParticleLightcone` class to extract requested
particles.

The code is parallelized with mpi4py and can be run as follows::

  # Location of the input lightcones
  basedir="./lightcones/"

  # Name of the lightcone to process
  basename="lightcone0"

  # Number of redshift bins to use
  nr_redshift_bins=4

  # HEALPix map resolution to use
  nside=32

  # HEALPix pixel ordering scheme
  order="nest"

  mpirun python3 -m mpi4py -m lightcone_io.index_particles \
                ${basedir} ${basename} ${nr_redshift_bins} ${nside} \
                ${outdir} --order ${order} --redshift-first

There is an example SLURM batch script to run on the FLAMINGO simulations on
COSMA-8 in ``scripts/FLAMINGO/sort_L1000N1800.sh``.

Computing halo membership in particle lightcones
------------------------------------------------

The module ``lightcone.particle_halo_ids`` can compute halo membership
for particles in the particle lightcone outputs. It works as follows:

  * The full halo lightcone is read in
  * For each halo in the halo lightcone we look up a mass and radius
    from SOAP (so SOAP must have been run on all snapshots)
  * The lightcone particles are read in
  * Particles within the radius of each halo in the halo lightcone are
    flagged as belonging to that halo
  * For each particle in the lightcone we write out the associated
    halo ID and mass

The mass and radius to use are specified by the name of the SOAP group
which they should be read from (e.g. ``--soap-so-name="SO/200_crit"``)
so it's possible to run the code using various halo radius
definitions.

Where the radii of several halos overlap there are three different
ways we can decide which halo to assign the particle to. These are
specified using the command line flag ``--overlap-method``. Possible
values are

  * ``fractional-radius``: for each particle we compute the distance
    to the halo centre in units of the halo radius. Particles are
    assigned to the halo for which this value is lowest.
  * ``most-massive``: particles within the radius of multiple halos
    are assigned to the most massive halo
  * ``least-massive``: particles within the radius of multiple halos
    are assigned to the least massive halo

This is also parallelized using mpi4py. To run it::

  # Location of the lightcone particle data
  lightcone_dir="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/particle_lightcones/"
  lightcone_base="lightcone0"

  # Format string to generate halo lightcone filenames
  halo_lightcone_filenames="/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/lightcone_halos/${lightcone_base}/lightcone_halos_%(file_nr)04d.hdf5"

  # Format string to generate SOAP catalogue filenames
  soap_filenames="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_%(snap_nr)04d.hdf5"

  # Directory to write the output to
  output_dir="/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/lightcone_particle_halo_ids/lightcone${lightcone_nr}/"

  mpirun python3 -m mpi4py -m lightcone_io.particle_halo_ids \
      "${lightcone_dir}" \
      "${lightcone_base}" \
      "${halo_lightcone_filenames}" \
      "${soap_filenames}" \
      "${output_dir}" \
       --soap-so-name="SO/200_crit" \
       --overlap-method=fractional_radius

There is a batch script to run this code on FLAMINGO on COSMA-8 in
``./scripts/FLAMINGO/halo_ids_L1000N1800.sh``.

.. _halo_lightcones:

Making halo lightcones
----------------------

An approximate halo lightcone can be constructed from a series of
snapshot halo catalogues by interpolating the halos between snapshots
to determine when the halo (or any periodic replication of the halo)
crosses the observer's lightcone. However, accurately interpolating
halo positions is difficult so instead we can use black hole particles
from the black hole particle lightcone output as tracers of the halo
positions.

Black holes are used as tracers because they are present in most well
resolved halos but less numerous than other particle types so they can
be output to high redshift without generating too much data to store.

The module ``lightcone_io.match_black_holes`` implements this. Each
simulation snapshot is assigned a redshift range which extends half
way to the next and previous snapshots. For each halo in the snapshot
we pick a black hole particle ID to trace the halo. Wherever this
particle appears in the particle lightcone, we place a copy of the
halo. This is done in such a way as to preserve the vector between the
halo's most bound particle and the chosen tracer particle.

The black hole tracer is chosen to be the most bound black hole which
also exists at the next and previous snapshots. This is intended to
minimize cases where a halo is lost because its black hole did not
exist at the time of lightcone crossing.

.. warning::
  This method has some significant drawbacks:
    * Evolution of the halos between the snapshot and the time of
      lightcone crossing is neglected, so the catalogue becomes less
      accurate at redshifts which are not close to a snapshot.
    * Halos with no black hole will be missing from the halo
      lightcone. This affects almost all halos below the black hole
      seeding halo mass. Black holes may also be lost from more massive
      halos (particularly satellite subhalos).

This module also uses mpi4py, so the command line to run it will be
along the lines of::

  mpirun -- python3 -m mpi4py -m lightcone_io.match_black_holes \
       "${halo_format}" ${first_sim_snap} ${last_sim_snap} \
        ${first_snap_to_process} {last_snap_to_process} \
        "${lightcone_dir}" "lightcone${lightcone_nr}" "${snapshot_format}" "${membership_format}" "${output_dir}" \
       --halo-type=HBTplus \
       --pass-through="InputHalos/IsCentral,InputHalos/NumberOfBoundParticles,BoundSubhalo/TotalMass,InputHalos/HBTplus/TrackId"

For descriptions of the command line parameters above, run::

  python3 -m lightcone_io.match_black_holes --help

There is an example script to run the code on FLAMINGO on Cosma-8 in
``scripts/FLAMINGO/match_bh_L1000N1800.sh``.
