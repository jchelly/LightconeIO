# Lightcone I/O for SWIFT

This is a python module for reading lightcone output from SWIFT simulations.

## Installation

The module can be installed to the user's home directory by running the
following in the source directory:
```
python ./setup.py install --user
```

## Reading lightcone HEALPix maps

When running with lightcones enabled, SWIFT can output HEALPix maps of various
quantities in concentric, spherical shells around the observer. The class
lightcone_io.healpix_maps.ShellArray is used to read these maps. The maps may
be in a single file per shell or split over many files.
```
import lightcone_io.healpix_maps as hm

# Location of the lightcone output
basedir="./lightcones/"

# Which lightcone to read
basename="lightcone0"

# Open a set of HEALPix maps
shell = hm.ShellArray(basedir, basename)
```
The ShellArray object is a sequence of Shells. Individual shells are accessed
by indexing with the shell number:
```
# Print the number of shells
print(len(shell))

# Print the inner and outer raddi of a single shell
print(shell[0].comoving_inner_radius, shell[0].comoving_outer_radius)
```
Each shell contains HEALPix maps of one or more quantities. These maps
are accessed with dictionary style indexing:
```
# Print names of the available maps for the first shell
print(list(shell[0]))

# Find the TotalMass map for the first shell
total_mass_map = shell[0]["TotalMass"]

# Return some information about this map
print(total_mass_map.nside) # HEALPix nside parameter
print(total_mass_map.dtype) # Data type of the pixel data
print(total_mass_map.units) # Units of the pixel data
print(len(total_mass_map))  # Total number of pixels
```
The pixel data can be read in by indexing the map. Simple [start:end] slices
can be used to read subsets of the pixels or Ellipses (...) can be used to
read all of the pixels:
```
# Read all of the pixels from the TotalMass map of the first shell
pixel_data = shell[0]["TotalMass"][...]

# Or read just the first 100 pixels
pixel_data_partial = shell[0]["TotalMass"][0:100]
```
If the unyt module is available then the results are returned as a unyt array
with unit information derived from the HDF5 attributes in the output files.

## Reading indexed lightcone particle outputs

Lightcone particle outputs can be post-processed to allow faster access to
specified areas of the sky and redshift ranges. These post-processed outputs
can be read with the class lightcone_io.particle_reader.IndexedLightcone:
```
import lightcone_io.particle_reader as pr

# Specify the name of one of the lightcone particle files
filename = "./lightcones/lightcone0_particles/lightcone0_0000.0.hdf5"

# Open the lightcone particle output
lightcone = pr.IndexedLightcone(filename)
```
The lightcone object acts like a dictionary where the particle types are
the keys. E.g. to see which types are available:
```
print(list(lightcone))
```
Particles can then be read in as follows:
```
# Quantities to read in
properties = ("Coordinates", "ParticleIDs")

# Position and angular radius (in radians) on the sky to read in
vector = (1., 0., 0.)
radius = np.radians(10.)

# Redshift range to read in
redshift_range = (0., 1.0)

# Read dark matter particles
data = lightcone["DM"].read(property_names, vector, radius, redshift_range)
```
The return value is a dictionary containing the quantities read in - in this
case Coordinates and ParticleIDs. If redshift_range=None then all redshifts
are read in. If vector=None and radius=None then the whole sky is read in.

Note that this may return particles outside the specified region because the
indexed lightcone is stored in chunks and all chunks overlapping the region
are returned.

There is also a way to iterate through the selected particles without reading
them all into memory:
```
for data in lightcone["DM"].iterate_chunks(property_names, vector, radius,
                                           redshift_range):
  pos = data["Coordinates"]
  ids = data["ParticleIDs"]
  # then do something with this subset of the selected particles...
```
Each iteration of this loop will receive a chunk of particles in the dict data.

## Combining HEALPix maps

The code above can read SWIFT HEALPix output regardless of how many files it
is split over. However, it may be desirable to reduce the number of files if
they're stored on a file system optimized for small numbers of large files.

The script bin/lightcone_io_combine_maps.py can be used to combine the maps 
for each shell into a single HDF5 file. This script is parallelized using 
mpi4py and can be run as follows:

```
input_dir=./lightcones/
output_dir=./indexed_lightcones/
nr_lightcones=2

mpirun python3 -m mpi4py \
    lightcone_io_combine_maps.py ${input_dir} ${nr_lightcones} ${output_dir}  
```

This will process all shells for the specified lightcones. The script assumes
that the lightcone basenames in the swift parameter file are lightcone0, 
lightcone1, ... etc.

There is an example SLURM batch script to run on the FLAMINGO simulations on
COSMA-8 in scripts/FLAMINGO/combine_L1000N1800.sh.

## Indexing particle outputs

SWIFT lightcone particle outputs are spread over many files and not sorted
in any useful order. The script bin/lightcone_io_index_particles.py can be
used to sort the particles and generate an index which can be used to
quickly retrieve particles by redshift and position on the sky.

The sky is divided into pixels using a low resolution HEALPix map and
each pixel is split into redshift bins. This defines a set of cells of
varying volume. The redshift bins are chosen such that the number of particles
per cell is roughly constant. The particles are then stored in order of which
cell they belong to and the location of each cell in the output files is
stored. This information is used by the lightcone_io.particle_reader module
to extract requested particles.

The script is parallelized with mpi4py and can be run as follows:
```
# Location of the input lightcones
basedir="./lightcones/"

# Name of the lightcone to process
basename="lightcone0"

# Number of redshift bins to use
nr_redshift_bins=4

# HEALPix map resolution to use
nside=32

mpirun python3 -m mpi4py lightcone_io_index_particles.py \
              ${basedir} ${basename} ${nr_redshift_bins} ${nside} ${outdir}
```
There is an example SLURM batch script to run on the FLAMINGO simulations on
COSMA-8 in scripts/FLAMINGO/sort_L1000N1800.sh.

Note that this script requires the virgo python module from
https://github.com/jchelly/VirgoDC .

## Example Scripts

### Plotting a pencil beam from the particle data

The script 'examples/plot_pencil_beam.py' reads in all particles in a 2 degree
radius about a vector along the x axis and makes a log scaled plot of projected
mass.

### Making a new HEALPix map from lightcone particle data

The script `examples/make_map_from_particles.py` shows how to make a new
HEALPix map by projecting particles from a lightcone particle output onto
the sky.

The script uses the iterate_chunks() method from the IndexedLightcone class
to read in chunks of particles in the required redshift range. It calculates
which HEALPix pixel each particle maps onto using healpy and adds the particle's
mass to that pixel. The resulting map is written to a new HDF5 file.


