#!/bin/env python

import h5py
import unyt
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.formats.swift as swift

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


class HaloCatalogue:
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

        # Convert to unyt arrays
        for name in data:
            units = swift.soap_units_from_attributes(data[name].attrs, reg)
            data[name] = unyt.unyt_array(data[name], units=units, registry=reg)
        
        return data

    
if __name__ == "__main__":

    halo_format = "/cosma8/data/dp004/jch/FLAMINGO/HBT/L1000N0900/HYDRO_FIDUCIAL/SOAP_uncompressed/HBTplus/halo_properties_{snap_nr:04d}.hdf5"
    first_snap = 2
    last_snap = 77
    cat = HaloCatalogue(halo_format, first_snap, last_snap)

    if comm_rank == 0:
        for snap_nr in range(first_snap, last_snap+1):
            print(snap_nr, cat.redshift[snap_nr])

    # Quantities we need to read in
    to_read = ("InputHalos/cofp", "InputHalos/index")
        
    data = cat.read(10, to_read)

    if comm_rank == 0:
        print(data)
