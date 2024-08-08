#!/bin/env python

import h5py
import numpy as np
import unyt
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.formats.swift as swift
import virgo.util.partial_formatter

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


class SOAPCatalogue:
    """
    Class to read SOAP halo catalogues for the halo lightcone
    """
    def __init__(self, halo_format, first_snap, last_snap):
        """
        Determine redshifts of all snapshots
        """
        self.redshift = {}
        if comm_rank == 0:
            for snap_nr in range(first_snap, last_snap+1):
                with h5py.File(halo_format.format(snap_nr=snap_nr), "r") as infile:
                    self.redshift[snap_nr] = float(infile["SWIFT/Cosmology"].attrs["Redshift"][0])
        self.redshift = comm.bcast(self.redshift)
        self.first_snap = first_snap
        self.last_snap = last_snap
        self.halo_format = halo_format
        self.description = {}
        
    def read(self, snap_nr, to_read):
        """
        Read in SOAP halos for the specified snapshot.

        Returns a dict of unyt arrays with units constructed from attributes.
        """
        
        # Create unit registry for this snapshot
        if comm_rank == 0:
            with h5py.File(self.halo_format.format(snap_nr=snap_nr), "r") as infile:
                reg = swift.soap_unit_registry_from_snapshot(infile["SWIFT"])
        else:
            reg = None
        reg = comm.bcast(reg)
                
        # Read the catalogue
        filename = self.halo_format.format(snap_nr=snap_nr)
        mf = phdf5.MultiFile(filename, file_idx=(0,), comm=comm)
        data = mf.read(to_read, read_attributes=True)

        # Store descriptions
        for name in data:
            if "Description" in data[name].attrs:
                self.description[name] = data[name].attrs["Description"]
        
        # Convert to unyt arrays
        for name in data:
            units = swift.soap_units_from_attributes(data[name].attrs, reg)
            data[name] = unyt.unyt_array(data[name], units=units, registry=reg)
        
        return data


class HBTplusCatalogue:
    """
    Class to read HBTplus halo catalogues for the halo lightcone

    This reads HBTplus halo properties and converts them to unyt arrays as if
    we read them from SOAP. This allows construction of lightcone halo
    catalogues without running SOAP on all (or any!) snapshots.
    """
    def __init__(self, halo_format, swift_format, first_snap, last_snap):
        """
        Determine redshifts of all snapshots and system of units
        """
        self.redshift = {}
        self.units = {}
        if comm_rank == 0:
            for snap_nr in range(first_snap, last_snap+1):
                with h5py.File(halo_format.format(snap_nr=snap_nr, file_nr=0), "r") as infile:
                    a = float(infile["Cosmology"]["ScaleFactor"][0])
                    self.redshift[snap_nr] = 1.0/a-1.0
                    if snap_nr == first_snap:
                        self.units["LengthInMpch"] = float(infile["Units"]["LengthInMpch"][0])
                        self.units["MassInMsunh"] = float(infile["Units"]["MassInMsunh"][0])
                        self.units["VelInKmS"] = float(infile["Units"]["VelInKmS"][0])
        self.redshift = comm.bcast(self.redshift)
        self.units = comm.bcast(self.units)
        self.first_snap = first_snap
        self.last_snap = last_snap
        self.halo_format = halo_format
        self.swift_format = swift_format
        self.description = {}
        
    def read(self, snap_nr, to_read):
        """
        Read in HBTplus halos for the specified snapshot.

        Returns a dict of unyt arrays with units constructed from attributes.
        Only supports a small subset of SOAP field names.
        """

        # Get the SWIFT unit registry for this snapshot
        if comm_rank == 0:
            filename = self.swift_format.format(snap_nr=snap_nr, file_nr=0)
            with h5py.File(filename, "r") as infile:
                registry = virgo.formats.swift.soap_unit_registry_from_snapshot(infile)
        else:
            registry = None
        registry = comm.bcast(registry)
        
        # Make a format string for the sub-files in this snapshot
        pf = virgo.util.partial_formatter.PartialFormatter()
        filenames = pf.format(self.halo_format, snap_nr=snap_nr, file_nr=None)

        # Read the full HBTplus catalogue
        mf = phdf5.MultiFile(filenames, file_nr_dataset="NumberOfFiles", comm=comm)
        subhalos = mf.read("Subhalos")

        # Compute halo indexes
        index_offset = comm.scan(len(subhalos)) - len(subhalos)
        index = np.arange(len(subhalos), dtype=int) + index_offset

        # Discard unresolved halos
        keep  = (subhalos["Nbound"] > 1)
        subhalos = subhalos[keep]
        index = index[keep]

        # Construct unyt representation of HBT's units
        h = unyt.Unit("h", registry=registry)
        swift_cmpc = unyt.Unit("a*swift_mpc", registry=registry)
        swift_msun = unyt.Unit("swift_msun", registry=registry)
        hbt_length_unit = (self.units["LengthInMpch"] * swift_cmpc/h).to(swift_cmpc)
        hbt_mass_unit = (self.units["MassInMsunh"] * swift_msun/h).to(swift_msun)
        
        # Extract the fields we want as unyt arrays using SOAP naming conventions
        data = {}
        for name in to_read:
            if name == "InputHalos/HaloCatalogueIndex":
                data[name] = unyt.unyt_array(index, units="dimensionless", registry=registry)
                self.description[name] = "Index of the halo in the original halo finder catalogue (first halo has index=0)"
            elif name == "InputHalos/HaloCentre":
                pos = subhalos["ComovingMostBoundPosition"].copy()
                data[name] = unyt.unyt_array(pos, units=hbt_length_unit, registry=registry)
                self.description[name] = "The centre of the halo as given by the halo finder"
            elif name == "InputHalos/IsCentral":
                is_central = np.where(subhalos["Rank"]==0, 1, 0)
                data[name] = unyt.unyt_array(is_central, units="dimensionless", registry=registry)
                self.description[name] = "Indicates if halo is central (1) or satellite (0)"
            elif name == "InputHalos/NumberOfBoundParticles":
                nbound = subhalos["Nbound"].copy()
                data[name] = unyt.unyt_array(nbound, units="dimensionless", registry=registry)
                self.description[name] = "Number of bound particles in the halo"
            elif name == "BoundSubhalo/TotalMass":
                mbound = subhalos["Mbound"].copy()
                data[name] = unyt.unyt_array(mbound, units=hbt_mass_unit, registry=registry)
                self.description[name] = "Mass of bound particles in the halo"
            elif name == "InputHalos/HBTplus/TrackId":
                trackid = subhalos["TrackId"].copy()
                data[name] = unyt.unyt_array(trackid, units="dimensionless", registry=registry)
                self.description[name] = "HBTplus TrackId of this halo"
            else:
                raise KeyError("Unsupported HBT property name!")
        
        return data


if __name__ == "__main__":

    #halo_format = "/cosma8/data/dp004/jch/FLAMINGO/HBT/L1000N0900/HYDRO_FIDUCIAL/SOAP_uncompressed/HBTplus/halo_properties_{snap_nr:04d}.hdf5"
    halo_format = "/cosma8/data/dp004/flamingo/Runs/L1000N0900/DMO_FIDUCIAL/HBT/{snap_nr:03d}/SubSnap_{snap_nr:03d}.{file_nr}.hdf5"
    swift_format = "/cosma8/data/dp004/flamingo/Runs/L1000N0900/DMO_FIDUCIAL/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5"
    first_snap = 2
    last_snap = 77
    #cat = SOAPCatalogue(halo_format, first_snap, last_snap)
    cat = HBTplusCatalogue(halo_format, swift_format, first_snap, last_snap)

    if comm_rank == 0:
        for snap_nr in range(first_snap, last_snap+1):
            print(snap_nr, cat.redshift[snap_nr])

    # Quantities we need to read in
    to_read = ("InputHalos/cofp", "InputHalos/index")
        
    data = cat.read(10, to_read)

    if comm_rank == 0:
        print(data)
