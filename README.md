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

## Combining HEALPix maps

The code above can read SWIFT HEALPix output regardless of how many files it
is split over. However, it may be desirable to reduce the number of files if
they're stored on a file system optimized for small numbers of large files.

The script bin/combine_maps_mpi.py can be used to combine the maps for each
shell into a single HDF5 file. This script is parallelized using mpi4py and
can be run as follows:

```
input_dir=./lightcones/
output_dir=./indexed_lightcones/
nr_lightcones=2

mpirun python3 -m mpi4py \
    combine_maps_mpi.py ${input_dir} ${nr_lightcones} ${output_dir}  
```

This will process all shells for the specified lightcones. The script assumes
that the lightcone basenames in the swift parameter file are lightcone0, 
lightcone1, ... etc.