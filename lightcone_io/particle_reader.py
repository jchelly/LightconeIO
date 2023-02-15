#!/bin/env python

import collections.abc

import numpy as np
import h5py
import re
import healpy as hp

try:
    import unyt
except ImportError:
    unyt=None
    raise ImportWarning("Unable to import unyt. Will not return unit info.")
    concatenate = np.concatenate
else:
    def units_from_attributes(dset):
        cgs_factor = dset.attrs["Conversion factor to CGS (not including cosmological corrections)"][0]
        U_I = dset.attrs["U_I exponent"][0]
        U_L = dset.attrs["U_L exponent"][0]
        U_M = dset.attrs["U_M exponent"][0]
        U_T = dset.attrs["U_T exponent"][0]
        U_t = dset.attrs["U_t exponent"][0]
        return cgs_factor * (unyt.A**U_I) * (unyt.cm**U_L) * (unyt.g**U_M) * (unyt.K**U_T) * (unyt.s**U_t) 
    concatenate = unyt.array.uconcatenate
    

def merge_cells(cell_offset, cell_length):

    keep = cell_length > 0
    cell_length = cell_length[keep]
    cell_offset = cell_offset[keep]
    
    ncells = len(cell_length)
    j = 0
    for i in range(1,ncells):
        if (cell_offset[i] == cell_offset[j]+cell_length[j]):
            cell_length[j] += cell_length[i]
            cell_length[i] = 0
        else:
            j = i

    keep = cell_length > 0
    cell_length = cell_length[keep]
    cell_offset = cell_offset[keep]

    return cell_offset, cell_length


class IndexedLightconeParticleType:
    """
    Class to read a single particle type from a lightcone
    """
    def __init__(self, type_name, metadata, index, units, filenames,
                 extra_filenames=None):

        self.type_name = type_name
        self.metadata  = metadata
        self.index     = index
        self.units     = units
        self.filenames = filenames
        self.extra_filenames = extra_filenames
        self.comm      = None
        self.comm_rank = 0
        self.comm_size = 1

        # Find range of particles in each file
        first_particle_in_file = index["first_particle_in_file"]
        num_particles_in_file  = index["num_particles_in_file"]
        last_particle_in_file  = first_particle_in_file + num_particles_in_file - 1

        # Find first particle in each cell
        first_particle_in_cell = index["cell_offset"]

        # Find which cell the first and last particles in each file belong to
        self.index["first_cell_in_file"] = (
            np.searchsorted(first_particle_in_cell, first_particle_in_file, side="right") - 1)
        self.index["last_cell_in_file"] = (
            np.searchsorted(first_particle_in_cell, last_particle_in_file, side="right") - 1)

        # Find which quantities we have for this type
        properties = {}
        for filename in self.filenames:
            with h5py.File(filename, "r") as infile:
                if type_name in infile:
                    for prop_name, dataset in infile[type_name].items():
                        if "a-scale exponent" in dataset.attrs:
                            shape = dataset.shape[1:]
                            dtype = dataset.dtype
                            if unyt is not None:
                                units = units_from_attributes(dataset)
                                properties[prop_name] = unyt.unyt_array(np.ndarray((0,)+shape, dtype=dataset.dtype), units)
                            else:
                                properties[prop_name] = np.ndarray((0,)+shape, dtype=dtype)
                            properties[prop_name].attrs = dict(dataset.attrs)
                    break
        self.properties = properties

        # Also check any extra files specified for additional quantities
        extra_properties = {}
        if extra_filenames is not None:
            assert len(filenames) == len(extra_filenames)
            for filename in self.extra_filenames:
                with h5py.File(filename, "r") as infile:
                    if type_name in infile:
                        for prop_name, dataset in infile[type_name].items():
                            if "a-scale exponent" in dataset.attrs:
                                shape = dataset.shape[1:]
                                dtype = dataset.dtype
                                if unyt is not None:
                                    units = units_from_attributes(dataset)
                                    extra_properties[prop_name] = unyt.unyt_array(np.ndarray((0,)+shape, dtype=dataset.dtype), units)
                                else:
                                    extra_properties[prop_name] = np.ndarray((0,)+shape, dtype=dtype)
                                extra_properties[prop_name].attrs = dict(dataset.attrs)
                        break
        self.extra_properties = extra_properties

    def set_mpi_mode(self, comm):
        self.comm = comm
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()

    def get_redshift_bins_in_range(self, redshift_min, redshift_max):
        """
        Return indexes of all redshift bins overlapping specified range
        """
        redshift_bins = self.index["redshift_bins"]
        z_bin_min = redshift_bins[:-1]
        z_bin_max = redshift_bins[1:]
        read_bin = (z_bin_min <= redshift_max) & (z_bin_max >= redshift_min)
        z_bin_index = np.arange(len(z_bin_min), dtype=int)[read_bin]
        return z_bin_index

    def get_pixels_in_radius(self, vector, radius):
        """
        Return indexes of all healpix pixels within radius of vector
        """
        nside = self.index["nside"]
        order = self.index["order"].decode() if "order" in self.index else "ring"
        if order == "nest":
            nest = True
        elif order == "ring":
            nest = False
        else:
            raise Exception("Invalid order parameter")
        pixels = hp.query_disc(nside, vector, radius, inclusive=True, nest=nest)
        pixels.sort()
        return pixels

    def get_cell_indexes_from_bins(self, redshift_bins, healpix_bins):
        """
        Given arrays of redshift and healpix bins to read, return
        indexes of all cells which must be read.
        """
        nr_pixels_to_read = len(healpix_bins)
        nr_redshifts_to_read = len(redshift_bins)
        nr_redshift_bins = len(self.index["redshift_bins"]) - 1
        nr_pixels = hp.pixelfunc.nside2npix(self.index["nside"])
        redshift_first = self.index["redshift_first"] if "redshift_first" in self.index else 0
        if redshift_first:
            redshift_bin_index = np.repeat(redshift_bins, nr_pixels_to_read)
            healpix_bin_index = np.tile(healpix_bins, nr_redshifts_to_read)
            cells_to_read = healpix_bin_index + (redshift_bin_index * nr_pixels)
        else:
            redshift_bin_index = np.tile(redshift_bins, nr_pixels_to_read)
            healpix_bin_index = np.repeat(healpix_bins, nr_redshifts_to_read)
            cells_to_read = redshift_bin_index + (healpix_bin_index * nr_redshift_bins)        
        return cells_to_read

    def read_cells(self, property_names, cells_to_read):
        """
        Given an array of cells to read, read the specified properties
        from the lightcone particle files.
        """

        # Check property names all exist
        for name in property_names:
            if name not in self.properties and name not in self.extra_properties:
                raise KeyError(f"Can't find property: {name}")

        # Check if we need to read from the extra data files
        read_extra = False
        for name in property_names:
            if name not in self.properties and name in self.extra_properties:
                read_extra = True

        # Cell indexes should be in ascending order
        assert np.all(cells_to_read[1:] > cells_to_read[:-1])

        # Find number of particles to be read in
        nr_particles = np.sum(self.index["cell_length"][cells_to_read], dtype=np.int64)
        nr_cells_to_read = len(cells_to_read)

        # Dict to store the output arrays (don't know their data types yet)
        data = {name : None for name in property_names}
        offset = {}

        # Find range of particles in each file
        first_particle_in_file = self.index["first_particle_in_file"]
        num_particles_in_file  = self.index["num_particles_in_file"]

        # Find range of particles in each cell
        first_particle_in_cell = self.index["cell_offset"]
        num_particles_in_cell  = self.index["cell_length"]

        # Find range of cells in each file
        num_cells = self.index["num_cells"]
        first_cell_in_file = self.index["first_cell_in_file"]
        last_cell_in_file  = self.index["last_cell_in_file"]

        # Flag cells which we want to read
        read_cell = np.zeros(num_cells, dtype=bool)
        read_cell[cells_to_read] = True

        # Loop over files in the lightcone
        for current_file, filename in enumerate(self.filenames):
            
            # Locate file with extra datasets, if any
            if self.extra_filenames is not None:
                extra_filename = self.extra_filenames[current_file]
            else:
                extra_filename = None

            # Skip files with no particles
            if num_particles_in_file[current_file] == 0:
                continue

            # Find range of cells which have particles in this file
            first_cell = first_cell_in_file[current_file]
            last_cell  = last_cell_in_file[current_file]

            # Get lengths and offset of these cells
            cell_offset = first_particle_in_cell[first_cell:last_cell+1]
            cell_length = num_particles_in_cell[first_cell:last_cell+1]

            # Discard cells which are empty or which will not be read
            keep = read_cell[first_cell:last_cell+1] & (cell_length > 0)
            cell_offset = cell_offset[keep]
            cell_length = cell_length[keep]

            # Merge adjacent cells
            cell_offset, cell_length = merge_cells(cell_offset, cell_length)

            # Skip files with no selected cells
            if sum(cell_length) == 0:
                continue

            # Make offsets relative to start of the current file
            cell_offset -= first_particle_in_file[current_file]

            # Open the file
            infile = h5py.File(filename, "r")
            if read_extra:
                extra_infile = h5py.File(extra_filename, "r")

            # Loop over quantities to read
            for name in property_names:

                # Find the dataset for this property
                if name in self.properties:
                    dset = infile[self.type_name][name]
                else:
                    dset = extra_infile[self.type_name][name]

                # Create output array, if we didn't already
                if data[name] is None:
                    shape = list(dset.shape)
                    shape[0] = nr_particles
                    if unyt is not None:
                        units = units_from_attributes(dset)
                        data[name] = unyt.unyt_array(np.ndarray(shape, dtype=dset.dtype), units)
                    else:
                        data[name] = np.ndarray(shape, dtype=dataset.dtype)
                    offset[name] = 0

                # Read the cells
                for (clen, coff) in zip(cell_length, cell_offset):
                    i1 = max((coff, 0))
                    i2 = min((coff+clen), num_particles_in_file[current_file])
                    num = i2 - i1
                    if num > 0:
                        data[name][offset[name]:offset[name]+num] = dset[i1:i2,...]
                        offset[name] += num

            # Close current file(s)
            infile.close()
            if read_extra:
                extra_infile.close()

        for name in property_names:
            assert offset[name] == nr_particles

        return data

    def get_cell_indexes_from_vector_radius_redshift(self, vector, radius, redshift_range):

        if redshift_range is not None:
            # Get redshift range to read
            redshift_min, redshift_max = [float(r) for r in redshift_range]
        else:
            # Read all redshifts
            redshift_min = self.index["redshift_bins"][0]
            redshift_max = self.index["redshift_bins"][-1]

        if vector is not None:
            # Select specified location on the sky
            if radius is None:
                raise ValueError("If specifying a radius must specify line of sight vector too")
            radius = float(radius)
            vector = np.asarray(vector, dtype=float)
        else:
            # Select full sky
            vector = np.asarray((1,0,0), dtype=float)
            radius = 2*np.pi

        # Find which redshift and healpix bins we should read in
        healpix_bins  = self.get_pixels_in_radius(vector, radius)
        redshift_bins = self.get_redshift_bins_in_range(redshift_min, redshift_max)
        cells_to_read = self.get_cell_indexes_from_bins(redshift_bins, healpix_bins)
        
        return cells_to_read

    def count_particles(self, vector=None, radius=None, redshift_range=None):

        cells_to_read = self.get_cell_indexes_from_vector_radius_redshift(vector, radius, redshift_range)
        if self.comm is not None:
            cells_to_read = self.split_cells_by_mpi_rank(cells_to_read)
        cell_size = self.index["cell_length"][cells_to_read]
        return np.sum(cell_size)

    def iterate_chunks(self, property_names, vector=None, radius=None, redshift_range=None,
                       max_particles=1048576):

        cells_to_read = self.get_cell_indexes_from_vector_radius_redshift(vector, radius, redshift_range)
        if self.comm is not None:
            cells_to_read = self.split_cells_by_mpi_rank(cells_to_read)
        nr_cells = len(cells_to_read)

        # Find the sizes of the selected cells
        cell_size = self.index["cell_length"][cells_to_read]

        # Loop over and yield chunks of cells containing up to max_particles
        i1 = 0
        while i1 < nr_cells:
            
            # Will always read at least one cell
            i2 = i1 + 1
            nr_to_read = cell_size[i1]

            # Determine how many more cells to read
            while i2 < nr_cells and nr_to_read+cell_size[i2] <= max_particles:
                nr_to_read += cell_size[i2]
                i2 += 1

            # Read and return the cell(s), if they contain any particles
            if nr_to_read > 0:
                yield self.read_cells(property_names, cells_to_read[i1:i2])

            # Advance to the nex set of cells
            i1 = i2

    def split_cells_by_mpi_rank(self, cells_to_read):

        # Find number of cells to read on each MPI rank
        nr_per_rank = np.zeros(self.comm_size, dtype=int)
        nr_per_rank[...] = len(cells_to_read) // self.comm_size
        nr_per_rank[:len(cells_to_read) % self.comm_size] += 1
        assert sum(nr_per_rank) == len(cells_to_read)

        # TODO: handle the case where some ranks have no data
        if np.any(nr_per_rank) == 0:
            raise RuntimeError("MPI rank was assigned no cells!")

        # Find offset to first cell on each rank
        offset_on_rank = np.cumsum(nr_per_rank) - nr_per_rank

        # Return range to read on this rank
        i1 = offset_on_rank[self.comm_rank]
        i2 = i1 + nr_per_rank[self.comm_rank]
        return cells_to_read[i1:i2]

    def read(self, property_names, vector=None, radius=None, redshift_range=None):

        # Read data for the selected cells
        cells_to_read = self.get_cell_indexes_from_vector_radius_redshift(vector, radius, redshift_range)
        if self.comm is not None:
            cells_to_read = self.split_cells_by_mpi_rank(cells_to_read)
        data = self.read_cells(property_names, cells_to_read)
        return data

    def read_exact(self, property_names, vector=None, radius=None, redshift_range=None):

        if vector is not None:
            vector = np.asarray(vector, dtype=float)

        # Filtering on angle is only implemented for radii < 90 degrees
        if radius is not None and radius > 0.5*np.pi:
            raise Exception("Angular radius must be <= pi/2 radians")

        # Make a dict to store the output.
        data = {name : [] for name in property_names}

        # Ensure we're reading the expansion factors if we need them
        keep_expansion = "ExpansionFactors" in property_names
        if redshift_range is not None and "ExpansionFactors" not in property_names:
            property_names = list(property_names) + ["ExpansionFactors",]

        # Ensure we're reading the coordinates if we need them
        keep_coordinates = "Coordinates" in property_names
        if radius is not None and "Coordinates" not in property_names:
            property_names = list(property_names) + ["Coordinates",]

        # Then we'll read the data in chunks and filter one chunk at a time
        max_particles = 10*1024*1024 # Use large-ish reads for efficiency
        for chunk in self.iterate_chunks(property_names, vector, radius, redshift_range, max_particles):

            # Find number of particles in this chunk
            nr_part = chunk[property_names[0]].shape[0]

            # Filter particles on redshift
            if redshift_range is not None:
                a_min = 1.0/(1.0+redshift_range[1])
                a_max = 1.0/(1.0+redshift_range[0])
                keep = (chunk["ExpansionFactors"]>=a_min) & (chunk["ExpansionFactors"]<=a_max)
            else:
                keep = np.ones(nr_part, dtype=bool)

            # Filter particles on radius
            if radius is not None:
                # Get normalized view direction vector
                norm_vector = np.asarray(vector, dtype=float)
                norm_vector = norm_vector / np.sqrt(np.sum(norm_vector**2))
                # Get 3D distance squared for each particle
                pos = chunk["Coordinates"]
                r2 = np.sum(pos**2, axis=1)
                # Find angle between view vector and each particle
                cos2_theta = (np.sum(vector[None,:]*pos, axis=1)**2)/r2
                keep = keep & (cos2_theta >= np.cos(radius)**2)
                del pos
                del r2
                del cos2_theta

            # Discard expansion factors if we were not asked to read them
            if keep_expansion == False and "ExpansionFactors" in chunk:
                del chunk["ExpansionFactors"]

            # Discard coordinates if we were not asked to read them
            if keep_coordinates == False and "Coordinates" in chunk:
                del chunk["Coordinates"]

            # Store all remaining properties
            for name in data:
                data[name].append(chunk[name][keep,...])
            del keep

        # Merge chunks
        for name in data:
            data[name] = concatenate(data[name])

        return data


class IndexedLightcone(collections.abc.Mapping):
    """
    Class used to read particle lightcones
    """
    def __init__(self, fname, comm=None, extra_filename=None):

        if comm is not None:
            comm_rank = comm.Get_rank()
            comm_size = comm.Get_size()
        else:
            comm_rank = 0
            comm_size = 1

        # In MPI mode rank 0 reads the metadata and then we broadcast it
        if comm_rank == 0:

            # Store base name so we can find the other files
            m = re.match(r"(.*)\.[0-9]+\.hdf5", fname)
            if m is not None:
                basename = m.group(1)
            else:
                raise IOError("Unable to extract base name from filename: %s" % fname)

            # Repeat for extra data files, if any
            if extra_filename is not None:
                m = re.match(r"(.*)\.[0-9]+\.hdf5", extra_filename)
                if m is not None:
                    extra_basename = m.group(1)
                else:
                    raise IOError("Unable to extract base name from filename: %s" % fname)

            with h5py.File(fname, "r") as infile:

                # Read metadata from the specified file
                metadata = {}
                for name in infile["Lightcone"].attrs:
                    metadata[name] = infile["Lightcone"].attrs[name]

                # Read units information
                units = {}
                units["Current"]     = infile["Units"].attrs["Unit current in cgs (U_I)"][0]
                units["Length"]      = infile["Units"].attrs["Unit length in cgs (U_L)"][0]
                units["Mass"]        = infile["Units"].attrs["Unit mass in cgs (U_M)"][0]
                units["Temperature"] = infile["Units"].attrs["Unit temperature in cgs (U_T)"][0]
                units["Time"]        = infile["Units"].attrs["Unit time in cgs (U_t)"][0]

                # Find names of all of the lightcone files
                filenames = []
                for i in range(metadata["nr_mpi_ranks"]):
                    filenames.append("%s.%d.hdf5" % (basename, i))

                # Find names of extra files
                if extra_filename is not None:
                    extra_filenames = []
                    for i in range(metadata["nr_mpi_ranks"]):
                        extra_filenames.append("%s.%d.hdf5" % (extra_basename, i))
                else:
                    extra_filenames = None

                # Determine particle types present
                self.particle_types = {}
                for type_name in infile["Cells"]:
                    index = {}
                    for name in infile["Cells"][type_name]:
                        index[name] = infile["Cells"][type_name][name][()]
                    self.particle_types[type_name] = IndexedLightconeParticleType(type_name, metadata, index, units, filenames,
                                                                                  extra_filenames)
        else:
            self.particle_types = None

        if comm is not None:
            self.particle_types = comm.bcast(self.particle_types)
            for pt in self.particle_types.values():
                pt.set_mpi_mode(comm)

    def __getitem__(self, key):
        return self.particle_types[key]

    def __iter__(self):
        for name in self.particle_types:
            yield name

    def __len__(self):
        return len(self.particle_types)
