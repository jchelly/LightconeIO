#!/bin/env python

import re
import io

import numpy as np

import h5py
import h5py.h5t
import h5py.h5s


def create_dataset(loc, name, type_id, shape, dcpl_id):
    """
    Create a new HDF5 dataset using supplied type and creation properties
    """    
    
    space_id = h5py.h5s.create_simple(shape, (h5py.h5s.UNLIMITED,)*len(shape))
    dset_id = h5py.h5d.create(loc.id, name.encode(), type_id, space_id, dcpl_id)
    space_id.close()
    dset_id.close()


def gather_lists(comm, l):
    l = comm.allgather(l)
    return [item for sublist in l for item in sublist]


class LightconeMetadata:

    def index_file_name(self, basedir=None):
        if basedir is None:
            basedir = self.basedir
        return "%(basedir)s/%(basename)s_index.hdf5" % {"basedir" : basedir, "basename" : self.basename}

    def particle_file_name(self, mpi_rank_nr, file_nr, basedir=None):
        if basedir is None:
            basedir = self.basedir

        # HACK for FLAMINGO L2800N5040 HYDRO_FIDUCIAL
        if basedir == "/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/lightcones/":
            if file_nr <= 80:
                basedir = "/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/L2800N5040/HYDRO_FIDUCIAL/lightcones/"

        return ("%(basedir)s/%(basename)s_particles/%(basename)s_%(file_nr)04d.%(mpi_rank_nr)d.hdf5" %
                {"basedir" : basedir, "basename" : self.basename,
                 "file_nr" : file_nr, "mpi_rank_nr" : mpi_rank_nr})

    def __init__(self, basedir, basename, comm=None):

        self.basedir = basedir
        self.basename = basename
        self.file = None

        if comm is None:
            comm_size = 1
            comm_rank = 0
        else:
            comm_rank = comm.Get_rank()
            comm_size = comm.Get_size()

        # Read the index to determine how many files there are
        if comm_rank == 0:
            with h5py.File(self.index_file_name(), "r") as infile:
                self.nr_mpi_ranks = infile["Lightcone"].attrs["nr_mpi_ranks"][0]
                self.final_particle_file_on_rank = infile["Lightcone"].attrs["final_particle_file_on_rank"]
                self.part_types = []
                for name in infile["Lightcone"].attrs:
                    m = re.match(r"minimum_redshift_(.*)", name)
                    if m:
                        self.part_types.append(m.group(1))
        else:
            self.nr_mpi_ranks = None
            self.final_particle_file_on_rank = None
            self.part_types = None
        if comm is not None:
            self.nr_mpi_ranks = comm.bcast(self.nr_mpi_ranks)
            self.final_particle_file_on_rank = comm.bcast(self.final_particle_file_on_rank)
            self.part_types = comm.bcast(self.part_types)

        # Generate the full list of filenames to read on this rank
        self.filenames = []
        count = 0
        for mpi_rank_nr in range(self.nr_mpi_ranks):
            for file_nr in range(self.final_particle_file_on_rank[mpi_rank_nr]+1):
                if count % comm_size == comm_rank:
                    filename = self.particle_file_name(mpi_rank_nr, file_nr)                
                    self.filenames.append(filename)
                count += 1
        self.nr_files = len(self.filenames)

        # Make an in-memory HDF5 file
        if comm_rank == 0:
            bio = io.BytesIO()
            tmpfile = h5py.File(bio, 'w')
            for ptype in self.part_types:
                tmpfile.create_group(ptype)

        # Determine number of particles of each type in each file
        self.nr_particles_file = {name : [] for name in self.part_types}
        for filename in self.filenames:
            with h5py.File(filename, "r") as infile:
                for ptype in self.part_types:
                    if ptype in infile:
                        nr = infile[ptype]["ExpansionFactors"].shape[0]
                    else:
                        nr = 0
                    self.nr_particles_file[ptype].append(nr)

        # In MPI mode we now need to combine the results from different ranks
        if comm is not None:            
            self.filenames = gather_lists(comm, self.filenames)
            for ptype in self.part_types:
                self.nr_particles_file[ptype] = gather_lists(comm, self.nr_particles_file[ptype])
            self.nr_files = len(self.filenames)

        # Calculate total number of particles of each type
        self.nr_particles_total = {name : 0 for name in self.part_types}
        for ptype in self.part_types:
            self.nr_particles_total[ptype] = sum(self.nr_particles_file[ptype])

        # Determine which quantities exist for each particle type
        if comm_rank == 0:
            self.properties = {}
            # Loop over particle types
            for ptype in self.part_types:
                # Loop over files to find one that contains particles of this type
                for i in range(self.nr_files):
                    if self.nr_particles_file[ptype][i] > 0:
                        # This file contains particles of this type
                        self.properties[ptype] = []
                        infile = h5py.File(self.filenames[i], "r")
                        for name in infile[ptype]:
                            dset = infile[ptype][name]
                            if "a-scale exponent" in dset.attrs:
                                # Create a zero sized in-memory dataset with the same data type
                                # and dataset creation property list. This preserves the filters,
                                # custom data types and and compression settings from the original
                                # dataset. Also copy any attributes.
                                dtype_id = dset.id.get_type()
                                dcpl_id = dset.id.get_create_plist()
                                shape = dset.shape[1:]
                                create_dataset(tmpfile[ptype], name, dtype_id, (0,)+shape, dcpl_id)
                                for attr_name in dset.attrs:
                                    tmpfile[ptype][name].attrs[attr_name] = dset.attrs[attr_name]
                                self.properties[ptype].append(name)
                        infile.close()
                        break
        else:
            self.properties = None
        if comm is not None:
            self.properties = comm.bcast(self.properties)

        # Store image of the in-memory file: this puts the information about special data types,
        # filters etc into a form that we can MPI_Bcast().
        if comm_rank == 0:
            tmpfile.close()
            self.file_image = bio.getvalue()
        else:
            self.file_image = None
        if comm is not None:
             self.file_image = comm.bcast(self.file_image)

        # Find offset to first particle in each file
        self.offset_file = {}
        for ptype in self.part_types:
            self.nr_particles_file[ptype] = np.asarray(self.nr_particles_file[ptype], dtype=np.int64)
            self.offset_file[ptype] = np.cumsum(self.nr_particles_file[ptype]) - self.nr_particles_file[ptype]

    def open_image(self):
        self.file = h5py.File(io.BytesIO(self.file_image), "r")

    def __del__(self):
        if self.file is not None:
            #self.file.close()
            self.file = None
