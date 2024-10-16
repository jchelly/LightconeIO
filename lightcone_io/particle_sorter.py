#!/bin/env python

import re
import os
import os.path
import shutil
import sys
import gc

import h5py

import numpy as np
import healpy as hp
from mpi4py import MPI

import virgo.mpi.parallel_sort as ps

import lightcone_io.particle_metadata as lm
import lightcone_io.memory_use as memory_use

import h5py.h5t
import h5py.h5s


class LightconeSorter:

    def message(self, message):
        if self.comm.Get_rank()==0:
            print(message)
        memory_use.report()
            
    def __init__(self, basedir, basename, comm, types=None):
        
        self.comm = comm
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()
        
        self.basedir = basedir
        self.basename = basename

        # Determine number of particles per file etc
        self.metadata = lm.LightconeMetadata(basedir, basename, comm)
        self.metadata.open_image()

        # Assign equally sized, contiguous ranges of particles to MPI ranks
        self.particle_offset = {}
        self.particle_count = {}
        for ptype in self.metadata.part_types:
            if self.metadata.nr_particles_total[ptype] > 0 and (types is None or ptype in types):
                nr_per_rank = self.metadata.nr_particles_total[ptype] // comm_size
                if nr_per_rank < 1:
                    raise Exception("Must have at least one particle per MPI rank!")
                self.particle_offset[ptype] = nr_per_rank * comm_rank
                self.particle_count[ptype] = nr_per_rank
                if comm_rank == comm_size-1:
                    self.particle_count[ptype] += (self.metadata.nr_particles_total[ptype] - nr_per_rank*comm_size)
                # Check total number of particles assigned is correct
                assert self.metadata.nr_particles_total[ptype] == self.comm.allreduce(self.particle_count[ptype])
            else:
                self.particle_count[ptype] = 0
                self.particle_offset[ptype] = 0

    def read_array(self, particle_type, property_name):
        """
        Read one particle property, distributed over all MPI ranks
        """

        # Allocate an array with the right data type and dimensions
        prop  = self.metadata.file[particle_type][property_name]
        dtype = prop.dtype.newbyteorder("=")
        shape = (self.particle_count[particle_type],)+prop.shape[1:]
        data  = np.ndarray(shape, dtype=dtype)

        # Find range of particles in each file
        offset_file = self.metadata.offset_file[particle_type]
        nr_particles_file = self.metadata.nr_particles_file[particle_type]

        # Loop over files
        output_offset = 0
        for file_nr in range(self.metadata.nr_files):
            
            # Find indexes i1:i2 of particles to read in this file
            i1 = self.particle_offset[particle_type] - offset_file[file_nr]
            i2 = i1 + self.particle_count[particle_type]
            if i1 < 0:
                i1 = 0
            if i2 > self.metadata.nr_particles_file[particle_type][file_nr]:
                i2 = self.metadata.nr_particles_file[particle_type][file_nr]
            
            # Read the data, if we have any particles to read from this file
            nread = i2-i1
            if nread > 0:
                with h5py.File(self.metadata.filenames[file_nr], "r") as infile:
                    data[output_offset:output_offset+nread,...] = infile[particle_type][property_name][i1:i2,...]
                output_offset += nread

        assert output_offset == self.particle_count[particle_type]
        return data

    def compute_bins(self, particle_type, nr_redshift_bins, nside, order,
                     redshift_first):
        """
        Decide on the redshift and healpix bins for this lightcone.

        Redshift bin edges are computed such that we have an approximately
        equal number of particles per bin.

        Returns an array of bins and a sorting index which puts the
        particles in order of which bin they belong to.
        """
        
        # HEALPix pixel ordering convention
        if order == "nest":
            nest=True
        elif order == "ring":
            nest=False
        else:
            raise Exception("Invalid order parameter")

        # Find which healpix pixel each particle is in
        self.message("  read coordinates")
        pos = self.read_array(particle_type, "Coordinates")
        self.message("  compute healpix pixel index")
        hp_bin_index = hp.pixelfunc.vec2pix(nside, pos[:,0], pos[:,1], pos[:,2], nest=nest)
        del pos
        
        # Get the redshift of each particle
        self.message("  compute redshift bin index")
        a = self.read_array(particle_type, "ExpansionFactors")
        assert len(a) == self.particle_count[particle_type]
        z = 1.0/a-1.0
        nr_particles_total = self.metadata.nr_particles_total[particle_type]

        # Get the full redshift range
        z_min = self.comm.allreduce(np.amin(z), op=MPI.MIN)
        z_max = self.comm.allreduce(np.amax(z), op=MPI.MAX)

        # Sort the redshifts
        z.sort()

        if nr_redshift_bins > 1:

            # Find global ranks of the particles at the bin edges
            # if we sort by redshift
            nr_per_bin = nr_particles_total // nr_redshift_bins
            rank = np.zeros(nr_redshift_bins-1, dtype=int)
            for i in range(nr_redshift_bins-1):
                rank[i] = (i+1)*nr_per_bin

            # Find the redshifts of these particles
            z_split, min_rank, max_rank = ps.find_splitting_points(z, rank, self.comm)

        elif nr_redshift_bins == 1:

            # Only one redshift bin, so no splitting points
            z_split = np.ndarray(0, dtype=float)

        else:
            raise Exception("Number of redshift bins must be positive!")
        del z

        # Make the full array of bin edges
        z_bins = np.ndarray(nr_redshift_bins+1)
        z_bins[0]    = z_min
        z_bins[1:-1] = z_split
        z_bins[-1]   = z_max

        # Find which redshift bin each particle is in
        z = 1.0/a-1.0 # Need un-sorted redshift here
        del a
        z_bin_index = np.searchsorted(z_bins, z, side="right")-1
        z_bin_index[z_bin_index==nr_redshift_bins] = nr_redshift_bins-1 # handle particle(s) with exactly z=z_max
        assert np.all((z >= z_bins[z_bin_index]) & (z <= z_bins[z_bin_index+1]))
        del z

        # Find number of pixels for this nside
        nr_pixels = hp.pixelfunc.nside2npix(nside)

        # Make an array of bins
        bin_dtype = np.dtype([("z_min", float),
                              ("z_max", float),
                              ("pixel", np.int64),
                              ("offset", np.int64),
                              ("length", np.int64)])
        bins = np.ndarray(nr_pixels*nr_redshift_bins, dtype=bin_dtype)
        offset = 0

        if redshift_first:
            # Sort bins by redshift, then by pixel within a redshift bin
            for bin_nr in range(nr_redshift_bins):
                for pixel_nr in range(nr_pixels):
                    bins[offset]["pixel"] = pixel_nr
                    bins[offset]["z_min"] = z_bins[bin_nr]
                    bins[offset]["z_max"] = z_bins[bin_nr+1]
                    bins[offset]["offset"] = 0
                    bins[offset]["length"] = 0
                    offset += 1
            bin_index = hp_bin_index + (z_bin_index * nr_pixels)
        else:
            # Sort bins by pixel, then by redshift within a pixel
            for pixel_nr in range(nr_pixels):
                for bin_nr in range(nr_redshift_bins):
                    bins[offset]["pixel"] = pixel_nr
                    bins[offset]["z_min"] = z_bins[bin_nr]
                    bins[offset]["z_max"] = z_bins[bin_nr+1]
                    bins[offset]["offset"] = 0
                    bins[offset]["length"] = 0
                    offset += 1
            bin_index = z_bin_index + (hp_bin_index * nr_redshift_bins)

        del z_bin_index
        del hp_bin_index

        # Count number of particles in each bin, summed over all MPI ranks
        bins["length"] = np.bincount(bin_index, minlength=len(bins))
        bins["length"] = self.comm.allreduce(bins["length"], op=MPI.SUM)

        # Compute offset to each bin if particles are sorted
        bins["offset"] = np.cumsum(bins["length"]) - bins["length"]

        # Generate a sorting index to put particles in order of bin
        self.message("  sort particles by bin index")
        sort_index = ps.parallel_sort(bin_index, comm=self.comm, return_index=True)

        # Return the array of bins and the sorting index
        return z_bins, bins, sort_index

    def write_sorted_lightcone(self, new_basedir, nr_redshift_bins, nside,
                               order="ring", redshift_first=True, lossy=True,
                               chunksize=0):
        """
        Write out a new lightcone with particles sorted by bin index
        """

        gc.collect()

        comm_rank = self.comm.Get_rank()
        comm_size = self.comm.Get_size()

        self.message("Copying index file")
        if comm_rank == 0:
            # Copy the file, if it doesn't already exist
            new_index_file_name = self.metadata.index_file_name(basedir=new_basedir)
            if not(os.path.exists(new_index_file_name)):
                os.makedirs(os.path.dirname(new_index_file_name), exist_ok=True)
                shutil.copyfile(self.metadata.index_file_name(), new_index_file_name)
            # Update number of files: will write one file per MPI rank
            with h5py.File(new_index_file_name, "r+") as new_index_file:
                new_index_file["Lightcone"].attrs["nr_mpi_ranks"] = (comm_size,)
                new_index_file["Lightcone"].attrs["final_particle_file_on_rank"] = np.zeros(comm_size, dtype=int)

        self.message("Creating output file")
        new_particle_file_name = self.metadata.particle_file_name(comm_rank, 0, new_basedir)
        os.makedirs(os.path.dirname(new_particle_file_name), exist_ok=True)
        outfile = h5py.File(new_particle_file_name, "w", libver=("v108", "latest"))
        outfile.create_group("Cells")

        with h5py.File(self.metadata.filenames[0], "r") as infile:
            # Copy metadata groups to the new output file
            infile.copy("Units", outfile)
            infile.copy("InternalCodeUnits", outfile)
            infile.copy("Lightcone", outfile)
            # Update some quantities
            outfile["Lightcone"].attrs["file_index"] = (0,)
            outfile["Lightcone"].attrs["mpi_rank"] = (comm_rank,)
            outfile["Lightcone"].attrs["nr_mpi_ranks"] = (comm_size,)
            del outfile["Lightcone"].attrs["expansion_factor"]

        # Loop over particle types
        for ptype in self.metadata.part_types:

            total_count = self.comm.allreduce(self.particle_count[ptype])
            if total_count > 0:

                self.message("Particle type %s (%d particles)" % (ptype, total_count))
                outfile.create_group(ptype)
                
                # Get sorting order and bins for this type
                z_bins, bins, sort_index = self.compute_bins(ptype, nr_redshift_bins, nside, order,
                                                             redshift_first)

                # Find offset to first particle in each file
                local_nr_particles = len(sort_index)
                local_particle_offset = self.comm.scan(local_nr_particles) - local_nr_particles 
                local_particle_offset = np.asarray(self.comm.allgather(local_particle_offset))
                local_nr_particles = np.asarray(self.comm.allgather(local_nr_particles))

                # Loop over quantities
                for name in self.metadata.properties[ptype]:

                    gc.collect()

                    # Find dataset template
                    dset_in = self.metadata.file[ptype][name]
                    
                    # Read the data for this particle property and sort into order of bin
                    self.message("  reading %s" % name)
                    data = self.read_array(ptype, name)
                    self.message("  sorting %s" % name)

                    if len(data.shape) == 1:
                        # Scalar quantity: just rearrange
                        ps.fetch_elements(data, sort_index, comm=self.comm, result=data)
                    elif len(data.shape) == 2:
                        # Vector quantity: save memory by rearranging one dimension at a time
                        for i in range(data.shape[1]):
                            col = data[:,i].copy()
                            ps.fetch_elements(col, sort_index, comm=self.comm, result=col)
                            data[:,i] = col
                            del col
                            gc.collect()
                    else:
                        raise Exception("Arrays with >2 dimensions not supported!")
                    self.message("  writing %s" % name)
                    dtype_id = dset_in.id.get_type()
                    dcpl_id = dset_in.id.get_create_plist()

                    # Disable lossy compression, if requested
                    if not(lossy):
                        try:
                            dcpl_id.remove_filter(h5py.h5z.FILTER_SCALEOFFSET)
                        except RuntimeError:
                            pass

                    # Override chunk size, if requested
                    if chunksize != 0:
                        chunks = (chunksize,)+(dset_in.shape[1:])
                        dcpl_id.set_chunk(chunks)

                    # Create the dataset
                    shape = (local_nr_particles[comm_rank],)+dset_in.shape[1:]
                    lm.create_dataset(outfile[ptype], name, dtype_id, shape, dcpl_id)
                    outfile[ptype][name][...] = data
                    del data
                    gc.collect()

                    # Add metadata to the array
                    dset_out = outfile[ptype][name]
                    for attr_name in dset_in.attrs:
                        dset_out.attrs[attr_name] = dset_in.attrs[attr_name]

                del sort_index
                gc.collect()

                # Update particle number for this type
                outfile["Lightcone"].attrs["cumulative_count_"+ptype] = (local_nr_particles[comm_rank],)

                # Write out binning information for this output
                cells = outfile["Cells"].create_group(ptype)
                cells["num_cells"] = len(bins)
                for name in bins.dtype.fields:
                    cells["cell_"+name] = bins[name]
                cells["first_particle_in_file"] = local_particle_offset
                cells["num_particles_in_file"]  = local_nr_particles
                cells["nside"] = nside
                cells["redshift_bins"] = z_bins
                cells["order"] = order
                cells["redshift_first"] = 1 if redshift_first else 0

        self.message("Done.")
        outfile.close()
