# Lightcone I/O for SWIFT

This is a python module for reading lightcone output from SWIFT simulations.

## Installation

The module can be installed to the user's home directory by running the
following in the source directory:
```
pip install --user .
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

If the simulation hasn't completed yet it wont be possible to initialize a
ShellArray because some of the files are missing. In that case you can open
individual shells. E.g.:
```
shell_nr = 10
shell = hm.Shell(basedir, basename, shell_nr)
total_mass_map = shell["TotalMass"][...]
```

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
You can use the properties attribute to see what quantities are available for
each particle type:
```
print(lightcone["Gas"].properties)
```
Each entry in properties is a zero element unyt array with the dtype, units
and shape of the quantity in the file and a `attrs` attribute which contains
a copy of the HDF5 attributes of the dataset.

Particles can be read in as follows:
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
are returned. There is also a read_exact() method which filters out the extra
particles. E.g.:

```
data = lightcone["DM"].read_exact(property_names, vector, radius, redshift_range)
```
This is likely to be slower because it's necessary to read in the coordinates for
spatial selection and the expansion factors for redshift selection even if
these quantities are not being returned. Computing the angles and redshifts
adds some CPU overhead too.

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

The module lightcone_io.combine_maps can be used to combine the maps for each
shell into a single HDF5 file. This code is parallelized using mpi4py and can
be run as follows:

```
input_dir=./lightcones/
output_dir=./indexed_lightcones/

mpirun python3 -m mpi4py -m lightcone_io.combine_maps.py \
       ${input_dir} ${output_dir} lightcone0 lightcone1 ...  
```

This will process all shells for the specified lightcones.

There is an example SLURM batch script to run on the FLAMINGO simulations on
COSMA-8 in scripts/FLAMINGO/combine_L1000N1800.sh.

## Indexing particle outputs

SWIFT lightcone particle outputs are spread over many files and not sorted
in any useful order. The module lightcone_io.index_particles can be
used to sort the particles and generate an index which can be used to
quickly retrieve particles by redshift and position on the sky.

The sky is divided into pixels using a low resolution HEALPix map and
each pixel is split into redshift bins. This defines a set of cells of
varying volume. The redshift bins are chosen such that the number of particles
per cell is roughly constant. The particles are then stored in order of which
cell they belong to and the location of each cell in the output files is
stored. This information is used by the lightcone_io.particle_reader module
to extract requested particles.

The code is parallelized with mpi4py and can be run as follows:
```
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

# Whether to sort first by pixel and then by redshift (0)
# or first by redshift then by pixel (1)
redshift_first=1

mpirun python3 -m mpi4py -m lightcone_io.index_particles \
              ${basedir} ${basename} ${nr_redshift_bins} ${nside} \
              ${order} ${redshift_first} ${outdir}
```
There is an example SLURM batch script to run on the FLAMINGO simulations on
COSMA-8 in scripts/FLAMINGO/sort_L1000N1800.sh.

Note that this script requires the virgo python module from
https://github.com/jchelly/VirgoDC .

## Computing halo membership in particle lightcones

The script `bin/lightcone_io_particle_halo_ids.py` can compute halo membership
for particles in the particle lightcone outputs. It works as follows:

  * The full halo lightcone is read in
  * For each halo in the halo lightcone we look up a mass and radius from SOAP
    (so SOAP must have been run on all snapshots)
  * The lightcone particles are read in
  * Particles within the radius of each halo in the halo lightcone are flagged
    as belonging to that halo
  * For each particle in the lightcone we write out the associated halo ID and mass

The mass and radius to use are specified by the name of the SOAP group
which they should be read from (e.g. `--soap-so-name="SO/200_crit"`)
so it's possible to run the code using various halo radius
definitions.

Where the radii of several halos overlap there are three different
ways we can decide which halo to assign the particle to. These are
specified using the command line flag `--overlap-method`. Possible
values are

  * `fractional-radius`: for each particle we compute the distance to the halo centre in units of the halo radius. Particles are assigned to the halo for which this value is lowest.
  * `most-massive`: particles within the radius of multiple halos are assigned to the most massive halo
  * `least-massive`: particles within the radius of multiple halos are assigned to the least massive halo

This is also parallelized using mpi4py. To run it:
```
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
```

There is a batch script to run this code on FLAMINGO on COSMA-8 in
./scripts/FLAMINGO/halo_ids_L1000N1800.sh.

## Example Scripts

### Plotting a HEALPix map

The script `examples/plot_healpix_map.py` shows how to read in a full HEALPix map
and plot it using the healpy mollview function.

### Plotting a pencil beam from the particle data

The script `examples/plot_pencil_beam.py` reads in all particles in a 2 degree
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

### Combining, correcting and downsampling FLAMINGO L1000N1800 maps on Cosma

To postprocess maps from a 1Gpc FLAMINGO run on Cosma:

  * Download and install this module
```
git clone git@github.com:jchelly/LightconeIO.git
module load python/3.10.1
cd LightconeIO
pip install --user .
```
  * Modify the output location (variable `output_dir`) in these scripts to point at a writable location:
```
scripts/FLAMINGO/combine_L1000N1800.sh
scripts/FLAMINGO/correct_L1000N1800.sh
scripts/FLAMINGO/downsample_L1000N1800.sh
```
  * Create a directory for the log files (the scripts will silently fail if this doesn't exist):
```
cd scripts/FLAMINGO
mkdir -p logs/L1000N1800
```
  * Submit the script to combine the maps into one file per shell. The job name (sbatch -J) specifies which simulation to process. The job array index (sbatch --array=...) is which lightcone(s) to do. E.g. for L1000N1800/HYDRO_FIDUCIAL:
```
sbatch -J HYDRO_FIDUCIAL --array=0-1 ./combine_L1000N1800.sh
```
  * Once that completes, the combined maps can be corrected:
```
sbatch -J HYDRO_FIDUCIAL --array=0-1 ./correct_L1000N1800.sh
```
  * And then the corrected map is downsampled:
```
sbatch -J HYDRO_FIDUCIAL --array=0-1 ./downsample_L1000N1800.sh
```
