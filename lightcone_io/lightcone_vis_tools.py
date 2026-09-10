#!/bin/env python
import sys
import numpy as np
import healpy as hp
import unyt
import math
import lightcone_io.healpix_maps as hm
import lightcone_io.particle_reader as pr
from lightcone_io.xray_utils import Snapshot_Cosmology_For_Lightcone
from lightcone_io.property_to_field_names import property_to_field, field_to_property
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Polygon, Rectangle, ConnectionPatch
import swiftsimio as sw
from swiftsimio.objects import cosmo_array
import collections
import inspect

def apply_expected_units(x, expected_units):
    """
    Convert input to expected units. 
    """
    # check if output spectrum has units
    if hasattr(x, "units") or isinstance(x, unyt.unyt_array):
        # if correct dimensions convert to expected units otherwise raise error 
        if x.units.dimensions==expected_units.dimensions:
            x.convert_to_units(expected_units)
        else:
            raise unyt.exceptions.UnytError(f"units do not match expected dimensions ({x.units}, {expected_units.units})")
    else:
        # apply units if no units found 
        if isinstance(x,  (list, np.ndarray)):
            x=unyt.unyt_array(x, units=expected_units)
        else:
            x=unyt.unyt_quantity(x, units=expected_units)
    return x

def round_down_10(x):
    return int(math.floor(x / 10) * 10)

def round_up_10(x):
    import math
    if x == 0:
        return 0
    power = 10 ** (int(math.floor(math.log10(abs(x)))) - 1)
    return math.ceil(x / power) * power

def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))



class BeamProjection:
    def __init__(self, vector, angular_diameter, redshift_range, cosmology=None, slice_thickness=None):
        """

        :param vector: direction vector as an array of 3 floats
        :type  vector: numpy.ndarray
        :param radius: angular diameter in degrees
        :type  radius: float
        :param redshift_range: redshift range to read
        :type  redshift_range: sequence of two floats [z_min, z_max]
        :param  cosmology: cosmology object of the simluation 
        :type   cosmology: astropy cosmology object or str
        """
        self.__beam_vec = tuple(vector)
        self.__diameter_deg=angular_diameter
        self.__radius = np.deg2rad(angular_diameter/2)
        self.__redshift_range=tuple(redshift_range)

        self.slice_thickness=apply_expected_units(slice_thickness, unyt.Mpc) # size on z-axis
        
        # try apply cosmology 
        if isinstance(cosmology, str):
            cosmology = Snapshot_Cosmology_For_Lightcone(cosmology).COSMO
        self.cosmology = cosmology

        self.__make_empty_flags()

    def __make_empty_flags(self,):
        # slice properties 
        
        self.in_slice_boolean={"Gas":None,"DM":None} #track particles being added to the mask 
        self.__gas_properties_added=None # track names of particle properties added to the slice. 
        self.__dm_properties_added=None # track names of particle properties added to the slice. 
        
        # boundaries of the plot in coordinate space
        self.__xmin = None
        self.__xmax = None
        self.__ymin = None
        self.__ymax = None
        self.__zmin = None
        self.__zmax = None
        
        self.axes_extent = None

        # snapshot filename used for mock data and cosmology
        self.__snapshot_filename = None

    def update_lc_particles(self, particle_data, particle_properties, slice_thickness, ptype):
        input_coords = particle_data["Coordinates"][:]
        # rotate the particles coordinates to produce consitant plots 
        rotated_coords = self.rotate_lc_coordinates(input_coords)
        del input_coords
        # make a mask that is True when particle is in the selected slice
        self.in_slice_boolean[ptype] = self.identify_particles_in_slice(rotated_coords, 1./particle_data["ExpansionFactors"].value - 1., slice_thickness)
        # update with 'in slice' mask 
        slice_particle_data={}
        slice_particle_data["Coordinates"]=rotated_coords[self.in_slice_boolean[ptype]]
        # keep particles and properties that are in the slice
        for k in particle_data.keys(): #for k in particle_properties:
            if k=="Coordinates":
                continue
            if k in particle_properties:
                slice_particle_data[k]=particle_data[k][self.in_slice_boolean[ptype]]

        nr_parts = len(particle_data["ExpansionFactors"])
        print("\n{n_all} particles in slice".format(n_all=nr_parts), flush=True)
        return slice_particle_data

    def rotate_lc_coordinates(self, coordinates):
        """
        Rotate particle coordinates so they are aligned with the vector (1,0,0)
        """
        if self.__beam_vec !=(1,0,0):
            rot_matrix = self.rotation_matrix_from_vectors(v_to=np.array([1., 0., 0.]))
            rotated_coordinates = coordinates @ rot_matrix.T
            return rotated_coordinates
        else:
            return coordinates
        
    def identify_particles_in_slice(self, coordinates, redshift, slice_thickness=10*unyt.Mpc):
        """
        Define simple mask for all partcles within slice, in terms of redshift and coordinates
        """
        dz=apply_expected_units(slice_thickness, unyt.Mpc)
        slice_midpoint = 0.5 * (np.max(coordinates[:,-1].to_value("Mpc")) + np.min(coordinates[:,-1].to_value("Mpc")))
        coords_boolean = (coordinates[:,-1].to_value("Mpc") >= slice_midpoint - dz.to_value("Mpc")/2) & (coordinates[:,-1].to_value("Mpc") <= slice_midpoint + dz.to_value("Mpc")/2)
        redshift_boolean = (redshift >= self.__redshift_range[0]) & (redshift <= self.__redshift_range[-1]) 
        return redshift_boolean & coords_boolean

    def add_property_to_slice(self, dset_name, dset, ptype):
        """
        Add additional properties to the slice particle data
        """
        if self.in_slice_boolean[ptype] is not None:
            if ptype == "Gas":
                if self.gas_particle_data is None:
                    print(f"No Gas particles in slice, cannot add {dset_name}")
                else:
                    if dset_name in self.gas_particle_data:
                        print(f"replacing {dset_name} in existing data")
                        del self.gas_particle_data[dset_name]
                    self.gas_particle_data[dset_name] = dset[self.in_slice_boolean[ptype]]
                    self.__gas_properties_added.append(dset_name)
            elif ptype == "DM":
                if self.dm_particle_data is None:
                    print(f"No DM particles in slice, cannot add {dset_name}")
                else:
                    if dset_name in self.dm_particle_data:
                        print(f"replacing {dset_name} in existing data")
                        del self.dm_particle_data[dset_name]
                    self.dm_particle_data[dset_name] = dset[self.in_slice_boolean[ptype]]
                    self.__dm_properties_added.append(dset_name)
        
        elif len(dset) != len(self.in_slice_boolean[ptype]):
            print(f"incorrect size of dataset")

        else:
            print(f"No {ptype} particles in slice, cannot add {dset_name}")

    def place_particles_in_slice(self, gas_particle_data,  gas_property_names, slice_thickness=None, dm_particle_data=None, dm_property_names=None, xy_buffer=10.):
    
        """
        Add particles in the slice to the lightcone

        :param  gas_particle_data:  gas particle data or a path to the particle lightcone
        :type   gas_particle_data:  lightcone_io.IndexedLightconeParticleType for 'Gas' or a str
        :param  gas_property_names: properties of gas particles to add to the slice. If None then essential properties are assumed 
                                        ["Coordinates", "ExpansionFactors","SmoothingLengths"]
        :type   gas_property_names: list
        :param  slice_thickness:    thickness of the slice [Mpc], length along the z-axis, of particles
        :type   slice_thickness:    int, float or unyt.unyt_quantity (units of Mpc or equivalent)
        :param  dm_particle_data:   dark matter particle data or a path to the particle lightcone that contains dark matter particles.
        :type   dm_particle_data:   lightcone_io.IndexedLightconeParticleType for 'DM' or a str
        :param  dm_property_names:  properties of dm particles to add to the slice. If dm_particle_data != None and dm_property_names is None, 
                                        all properties are loaded by default ["Coordinates","ExpansionFactors"]
        :type   dm_property_names:  list
        :param  xy_buffer:          distance [Mpc] between the min and max position of a particle and the boundary of the slice.
        :type   xy_buffer:          float or int
        """ 
        
        # update units for the size of the slice 
        if slice_thickness is None:
            slice_thickness=apply_expected_units(self.slice_thickness, unyt.Mpc)
        else:
            slice_thickness=apply_expected_units(slice_thickness, unyt.Mpc)
        
        if gas_particle_data is not None:
            # add required properties where needed
            required_properties = ["Coordinates", "ExpansionFactors","SmoothingLengths"]
            if gas_property_names is None:
                slice_gas_properties=required_properties
            else:
                slice_gas_properties=[prop for prop in gas_property_names]
                for required_prop in required_properties:
                    if required_prop not in slice_gas_properties:
                        slice_gas_properties.append(required_prop)
                
            self.__gas_properties_added = gas_property_names

            # check if particle data has to be loaded
            if isinstance(gas_particle_data, str)==False:
                # sanity check properties against provided data
                for prop in self.__gas_properties_added:
                    if prop not in gas_particle_data:
                        raise ValueError(f"{prop} not in Gas particle data")
            
            elif isinstance(gas_particle_data, str):
                print(f"\nLoading gas particle data ....")
                # Open the lightcone
                lightcone = pr.IndexedLightcone(gas_particle_data)
                # read particle data from lightcone
                gas_particle_data = lightcone['Gas'].read(
                    property_names=self.__gas_properties_added,
                    redshift_range=self.__redshift_range,
                    vector=self.__beam_vec, radius=self.__radius
                )
                nr_parts = len(gas_particle_data["ExpansionFactors"])
                print("\nread in {n_all} Gas particles".format(n_all=nr_parts))
            
            self.gas_particle_data = self.update_lc_particles(gas_particle_data, self.__gas_properties_added, slice_thickness, "Gas")
            
            #set limits of of the beam 
            self.__lightcone_limits(self.gas_particle_data["Coordinates"], axis_boundaries_buffer=xy_buffer) 

        else:
            self.gas_particle_data=None
        
        if dm_particle_data is not None: 
            
            # add required properties where needed
            required_properties = ["Coordinates", "ExpansionFactors"]
            if dm_property_names is None:
                slice_dm_properties=required_properties
            else:
                slice_dm_properties=[prop for prop in dm_property_names]
                for required_prop in required_properties:
                    if required_prop not in slice_dm_properties:
                        slice_dm_properties.append(required_prop)

            self.__dm_properties_added=slice_dm_properties
            # check if particle data has to be loaded
            if isinstance(dm_particle_data, str)==False:
                # sanity check properties against provided data
                for prop in self.__dm_properties_added:
                    if prop not in dm_particle_data:
                        raise ValueError(f"{prop} not in DM particle data")
            
            elif isinstance(dm_particle_data, str):
                print(f"\nLoading DM particle data ....")
                # Open the lightcone
                lightcone = pr.IndexedLightcone(dm_particle_data)
                # read particle data from lightcone
                dm_particle_data = lightcone['DM'].read(
                    property_names=self.__dm_properties_added,
                    redshift_range=self.__redshift_range,
                    vector=self.__beam_vec, radius=self.__radius
                )
            dm_nr_parts = len(dm_particle_data["ExpansionFactors"])
            print("\nread in {n_all} DM particles".format(n_all=dm_nr_parts))
            self.dm_particle_data = self.update_lc_particles(dm_particle_data, self.__dm_properties_added, slice_thickness, "DM")
            if self.axes_extent is None:
                self.__lightcone_limits(self.dm_particle_data["Coordinates"], axis_boundaries_buffer=xy_buffer) 
        else:
            self.dm_particle_data=None

    def __lightcone_limits(self, coordinates, axis_boundaries_buffer=10):
        self.__xmin = np.amin(coordinates[:,0])
        self.__xmax = np.amax(coordinates[:,0])
        self.__ymin = np.amin(coordinates[:,1])
        self.__ymax = np.amax(coordinates[:,1])
        self.__zmin = np.amin(coordinates[:,1])
        self.__zmax = np.amax(coordinates[:,1])

        self.axes_extent = [
            self.__xmin.to_value("Mpc")+axis_boundaries_buffer/2,
            self.__xmax.to_value("Mpc")+axis_boundaries_buffer/2,
            self.__ymin.to_value("Mpc"),
            self.__ymax.to_value("Mpc")
        ]
        self.axis_buffer=axis_boundaries_buffer

    def project_properties(self, project_particle_properties, snapshot_filename=None, resolution=1024, assign_units=None, ptype="Gas"):
        """
        Returns a 2D histogram of the selected properties for a given particle type
        """

        # export to snapshot datatype to project with swiftsimio
        use_lc_properties = [prop for prop in project_particle_properties]

        if self.cosmology is None:
            self.cosmology=Snapshot_Cosmology_For_Lightcone(snapshot_filename).COSMO
            # update snapshot name stored if used for cosmology 
            self.__snapshot_filename = snapshot_filename
        
        mock_snap_data, preffered_units = self.make_mock_snapshot(use_lc_properties, snapshot_filename, resolution=1024, assign_units=None, ptype=ptype)
        
        
        #create region to project over
        plot_region= cosmo_array(
                [
                    self.__xmin.to_value("Mpc")-self.axis_buffer, self.__xmax.to_value("Mpc")+self.axis_buffer/2, 
                    self.__ymin.to_value("Mpc")-self.axis_buffer/2, self.__ymax.to_value("Mpc")+self.axis_buffer/2
                    ],
                unyt.Mpc,
                comoving=True,
                scale_factor=1.,  # a=0.5, i.e. z=1
                scale_exponent=1,  # distances scale as a**1, so the scale exponent is 1
            )
        
        projection_outputs=[]
        if ptype=="Gas":
            for prop in project_particle_properties:
                field_name = property_to_field[prop] if prop in property_to_field else prop # allow for custom properties 
                print(f"Projecting:\t{prop} [{field_name}]")
                proj = sw.visualisation.projection.project_pixel_grid(
                    mock_snap_data.lightcone_gas,
                    resolution=resolution,
                    project=field_name,
                    parallel=False,
                    periodic=True,
                    region=plot_region,
                    )

                projection_outputs.append(proj)
        
        elif ptype=="DM":
            for prop in project_particle_properties:
                field_name = property_to_field[prop]
                print(f"Projecting:\t{prop} [{field_name}]")
                proj = sw.visualisation.projection.project_pixel_grid(
                    mock_snap_data.lightcone_dm,
                    resolution=resolution,
                    project=field_name,
                    parallel=False,
                    periodic=True,
                    region=plot_region,
                    )

                projection_outputs.append(proj)

        return projection_outputs

    def add_lc_gas_to_snap(self, snap, particle_properties, assign_units):
 
        if hasattr(snap, "lightcone_gas"):
            raise ValueError("Snapshot already has mock snapshot gas particles!!!")

        # define mock fields
        mock_fields=["metadata"] 
        # re-name for swiftsim_io
        for prop in particle_properties:
            if prop in property_to_field:
                mock_fields.append(property_to_field[prop])
            else:
                mock_fields.append(prop)
        
        # add required fields
        for i, required_prop in enumerate(["Coordinates","SmoothingLengths"]): 
            if required_prop not in particle_properties and property_to_field[required_prop] not in mock_fields:
                mock_fields.append(property_to_field[required_prop])
                particle_properties.append(required_prop)

        # determine the preferred units of each property 
        preffered_units={}
        for idx, prop in enumerate(particle_properties):

            if prop == "Coordinates":
                unit_str = "Mpc" 
            elif prop== "SmoothingLengths":
                unit_str="Mpc"
            elif prop == "Masses":
                unit_str="Msun"
            else:
                print(prop)
                if assign_units is None:
                    print()
                    unit_str = str(self.gas_particle_data[prop].units.expr)
                else:
                    unit_str=assign_units[idx]
            
            preffered_units[prop] = unit_str


        MockTotal = collections.namedtuple("MockTotal", mock_fields)
        
        lc_coordinates = self.__lc2cosmoarray(self.gas_particle_data, "Coordinates", "Mpc")
        #lc_masses = self.__lc2cosmoarray(self.gas_particle_data, "Masses", "Msun")
        lc_smoothing_lengths = self.__lc2cosmoarray(self.gas_particle_data, "SmoothingLengths", "Mpc")

        mock_values = {
            "metadata":snap.metadata,
            "coordinates":lc_coordinates,
            #"masses":lc_masses,
            "smoothing_lengths":lc_smoothing_lengths,
        }
        
        for idx, field_name in enumerate(mock_fields):
            if field_name in ["metadata", "coordinates", "smoothing_lengths"]:
                continue 
            #print(idx, field_name, field_to_property[field_name])
            prop_name=field_to_property[field_name] if field_name in field_to_property else field_name
            mock_values[field_name] = self.__lc2cosmoarray(self.gas_particle_data, prop_name, preffered_units[prop_name])
        
        snap.lightcone_gas =  MockTotal(**mock_values)
        
        del particle_properties
        
        return snap, preffered_units

    def add_lc_dm_to_snap(self, snap, particle_properties, assign_units):
 
        if hasattr(snap, "lightcone_dm"):
            raise ValueError("Snapshot already has mock snapshot DM particles!!!")


        # define mock fields
        mock_fields=["metadata"] 
        # re-name for swiftsim_io
        for prop in particle_properties:
            if prop in property_to_field:
                mock_fields.append(property_to_field[prop])
            else:
                mock_fields.append(prop)
        
        # add required fields
        for i, required_prop in enumerate(["Coordinates","Masses","SmoothingLengths"]): 
            if required_prop not in particle_properties and property_to_field[required_prop] not in mock_fields:
                mock_fields.append(property_to_field[required_prop])
                particle_properties.append(required_prop)


        preffered_units={}

        for idx, prop in enumerate(particle_properties):
            
            if prop == "Coordinates":
                unit_str = "Mpc" 
            elif prop== "SmoothingLengths":
                unit_str="Mpc"
            elif prop == "Masses":
                unit_str="Msun"
            else:
                print(prop)
                if assign_units is None:
                    print()
                    unit_str = str(self.dm_particle_data[prop].units.expr)
                else:
                    unit_str=assign_units[idx]
            
            preffered_units[prop] = unit_str

        lc_coordinates = self.__lc2cosmoarray(self.dm_particle_data, "Coordinates", "Mpc")
        lc_masses = self.__lc2cosmoarray(self.dm_particle_data, "Masses", "Msun")

        # compute smoothing lengths for dm particles. 
        # 1) shift position of particles so they are inside the snapshot box and 2) generate smoothing lengths
        shifted_coords=unyt.unyt_array(np.zeros_like(self.dm_particle_data["Coordinates"]), units=unyt.Mpc)
        ax_sidelengths=unyt.unyt_array(np.zeros(3, dtype=float), units=unyt.Mpc)
        for i in range(3):
            ax_min=np.min(self.dm_particle_data["Coordinates"][:, i].to_value("Mpc"))
            if  ax_min< 0: # shift partciles on axes to only have positive values
                shifted_coords[:, i]+=self.dm_particle_data["Coordinates"][:, i].to_value("Mpc") - ax_min
                #print(x.min(), x.max())
            else:
                shifted_coords[:, i]+=self.dm_particle_data["Coordinates"][:, i].to_value("Mpc")

            if np.max(shifted_coords[:, i].to_value("Mpc")) < snap.metadata.boxsize[i].to_value("Mpc"):
                ax_sidelengths[i] =snap.metadata.boxsize[i].to_value("Mpc")
            else:
                ax_sidelengths[i] = round_up_10(np.max(shifted_coords[:, i].to_value("Mpc"))+0.1) # new box sidelengths go from 0-> max part location in lc


        # shifted coordinates 
        lc_coordinates_shifted = cosmo_array(
            shifted_coords.to_value("Mpc"),
            unyt.Mpc,
            comoving=True,
            scale_factor=1.,  # a=0.5, i.e. z=1
            scale_exponent=1,  # distances scale as a**1, so the scale exponent is 1
        )
        # define box sidelength for the lightcone 
        lc_sidelength = cosmo_array(
            ax_sidelengths.to_value("Mpc"),
            unyt.Mpc,
            comoving=True,
            scale_factor=1.,  # a=0.5, i.e. z=1
            scale_exponent=1,  # distances scale as a**1, so the scale exponent is 1
        )

        lc_smoothing_lengths = sw.visualisation.generate_smoothing_lengths(
           lc_coordinates_shifted,
           lc_sidelength,
           kernel_gamma=1.8,
           neighbours=32, #57
           speedup_fac=2,
           dimension=3,
       )

        #lc_smoothing_lengths = self.__lc2cosmoarray(self.dm_particle_data, "SmoothingLengths", "Mpc")

        MockTotal = collections.namedtuple("MockTotal", mock_fields)
        
        mock_values = {
            "metadata":snap.metadata,
            "coordinates":lc_coordinates,
            "masses":lc_masses,
            "smoothing_lengths":lc_smoothing_lengths,
        }
        
        for idx, field_name in enumerate(mock_fields):
            if field_name in ["metadata", "coordinates", "masses", "smoothing_lengths"]:
                continue 
            #print(idx, field_name, field_to_property[field_name])
            prop_name=field_to_property[field_name]
            mock_values[field_name] = self.__lc2cosmoarray(self.dm_particle_data, prop_name, preffered_units[prop_name])
        
        snap.lightcone_dm =  MockTotal(**mock_values)
        
        return snap, preffered_units

    def make_mock_snapshot(self, particle_properties, snapshot_filename, resolution=1024, assign_units=None, ptype="Gas"):
        
        # sanity check properties
        for prop in particle_properties:

            if ptype=="Gas":
                if prop not in self.__gas_properties_added:
                    raise ValueError(f"{prop} values not in slice")
                else:
                    continue 
            elif ptype=="DM":
                if prop!="SmoothingLengths" and prop not in self.__dm_properties_added:
                    raise ValueError(f"{prop} values not in slice")
                else:
                    continue 
            else:
                raise ValueError("unknown ptype")

        # load a spatially constrained subset of snapshot particles to then replace with lightcone data
        mask = sw.mask(snapshot_filename)
        # The full metadata object is available from within the mask
        boxsize = mask.metadata.boxsize
        load_region = [[1 * b/b.value, 2 * b/b.value] for b in boxsize]
        # Spatially constrain and load the snapshot
        mask.constrain_spatial(load_region)
        snap = sw.load(snapshot_filename, mask=mask)

        # create mock snapshot gas particle data from the lightcone particles 
        if ptype=="Gas":
            snap, preffered_units = self.add_lc_gas_to_snap(snap, particle_properties, assign_units)

        elif ptype=="DM":
            snap, preffered_units = self.add_lc_dm_to_snap(snap, particle_properties, assign_units)

        return snap, preffered_units

    @staticmethod
    def __lc2cosmoarray(particle_data, property_name, property_units):
        return cosmo_array(
            particle_data[property_name].to_value(property_units),
            particle_data[property_name].to(property_units).units,
            comoving=True, # assume comoving for all lightcone properties 
            scale_factor=1., scale_exponent=1 # assume defined at z=0 and has no additional scale factor effects
            )

    def rotation_matrix_from_vectors(self, v_to=np.array([1., 0., 0.])):
        """
        Rotation matrix that rotates v_from onto v_to.
        """
        v_from=np.array([self.__beam_vec[0], self.__beam_vec[1], self.__beam_vec[2]])
        a = np.asarray(v_from, dtype=float)
        b = np.asarray(v_to, dtype=float)

        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        v = np.cross(a, b)
        c = np.dot(a, b)

        # vectors are parallel
        if np.isclose(c, 1):
            return np.eye(3)

        # vectors are anti-parallel
        if np.isclose(c, -1):
            # choose any axis perpendicular to a
            axis = np.cross(a, [1, 0, 0])
            if np.linalg.norm(axis) < 1e-10:
                axis = np.cross(a, [0, 1, 0])
            axis /= np.linalg.norm(axis)

            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])

            return np.eye(3) + 2 * K @ K

        s = np.linalg.norm(v)

        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])

        R = np.eye(3) + K + K @ K * ((1 - c) / s**2)

        return R

    def filter_kwargs(self, func, kwargs):
        params = inspect.signature(func).parameters
        return {k: v for k, v in kwargs.items() if k in params}
    
    def split_beam_plot(self, numb_wedges, projection_data, colour_maps, 
            axs=None, filename=None,
            angular_diameter=None, cosmology=None, redshift_range=None, axes_extent=None, update_badcol=True, figsize=(7,7), titles=None, **kwargs):
        
        """
        Create plot of the whole beam, split into seperate wedges. 

        Returns list of each projected wedge.  

        :param  numb_wedges:        number of wedges or different projections to show in the beam
        :type   numb_wedges:        int
        :param  projection_data:    nested list containing different segements of the beam the 2D array to plot and 
                                        a tuple with the min and max values shown in the img [2D array, (min, max)]
        :type   projection_data:    list
        :param  colour_maps:        list of colour maps for each segment of the beam
        :type   colour_maps:        list  
        :param  axs:                Default=None, If None create a new axes for the plot, otherwise plot onto the axes given 
        :type   axs:                matplotlib.axes._axes.Axes
        :param  filename:           None, if given, write plot to this file 
        :type   filename:           str
        :param  angular_diameter:   total angular diameter of beam in degrees
        :type   angular_diameter:   float
        :param  cosmology:          the simualtions cosmology model
        :type   cosmology:          astropy comsology object, astropy.cosmology.flrw.w0wacdm.w0waCDM
        :param  redshift_range:     maximum and minimum redshift of the beam shown
        :type   redshift_range:     list, np.ndarray or tuple 
        :param  update_badcol:      If true, modify all colour maps so that the minimum, nan and None values are set to black 
        :type   update_badcol:      boolean
        :param  axes_extent:        Extent of the plots axes, in coordinate space. 
                                    If None, then use the extent defined by the coordinates of the particles added to the slice. 
        :param  kwargs:             All additional arguments to be passed onto the add_wedge and add_beam_axes functions. 
        """

        # update slice information and call predefined values where needed
        if redshift_range is None:
            if self.__redshift_range is None:
                raise ValueError("Redshift range is not defined")
        else:
            self.__redshift_range = redshift_range
        
        if axes_extent is None:
            if self.axes_extent is None:
                raise ValueError("Axes range is not defined")
        else:
            self.axes_extent = axes_extent

        if angular_diameter is None:
            if self.__diameter_deg is None:
                raise ValueError("angular diameter is not defined")
        else:
            self.__diameter_deg = angular_diameter

        if cosmology is None:
            if self.cosmology is None:
                if self.__snapshot_filename is None:
                    raise ValueError("Cosmology is not defined")
                else:
                    self.cosmology=Snapshot_Cosmology_For_Lightcone(self.__snapshot_filename).COSMO
        else:
            self.cosmology=cosmology

        #define wedge angles
        beam_ang_offset_deg = 0. #01
        beam_radius_deg=(self.__diameter_deg-beam_ang_offset_deg)/2
        wedge_diameter_deg= (self.__diameter_deg-beam_ang_offset_deg)/numb_wedges # angular diameter of one wedge
        print(f"beam angular diameter:\t{self.__diameter_deg:.2f} [deg]", flush=True)
        print(f"wedge angular diameter:\t{wedge_diameter_deg:.2f} [deg]", flush=True)

        # create redshift ticks for the range given 
        all_redshift_major_ticks=np.arange(0.05, 5.0, 0.05)
        all_redshift_minor_ticks=np.arange(0.01, 5.0, 0.01)

        # refine range of redshift ticks for the redshift range given 
        if self.__redshift_range[0] < 0.01:
            self.__redshift_range[0]=0.01
        redshift_axis_ticks=np.concatenate((all_redshift_major_ticks[(all_redshift_major_ticks>self.__redshift_range[0])&(all_redshift_major_ticks<self.__redshift_range[-1])], [self.__redshift_range[-1]]))
        redshift_axis_minor_ticks=np.concatenate((all_redshift_minor_ticks[(all_redshift_minor_ticks>self.__redshift_range[0])&(all_redshift_minor_ticks<self.__redshift_range[-1])], [self.__redshift_range[-1]]))

        # define inner and outer radius
        rmax=self.cosmology.comoving_distance(redshift_axis_ticks[-1]).to_value("Mpc")
        rmin=self.cosmology.comoving_distance(redshift_axis_minor_ticks[0]).to_value("Mpc")

        wedge_imgs=[[] for i in range(numb_wedges)]

        # if not adding to other plot, then create empty subplot
        
        if axs is None:
            return_ax = False
            return_fig = True
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, gridspec_kw={'height_ratios': [1], 'width_ratios': [1], 'hspace': 0, 'wspace': 0})
        else:
            return_ax = True
            return_fig = False

        axs.margins(0)
        axs.set_aspect("equal")
        axs.axis("off")

        if titles is not None:
            assert len(titles) == numb_wedges
            
        # seperate kwargs for image function and axes function 
        img_kwargs = self.filter_kwargs(self.add_wedge, kwargs)
        axes_kwargs = self.filter_kwargs(self.add_beam_axes, kwargs)

        # iterate through the different split beams and add to plot
        for wedge_idx in range(numb_wedges):
            
            wedge_cmap = plt.colormaps[colour_maps[wedge_idx]].copy()
            
            if update_badcol==True:
                # update colour maps to ensure bad colour = 'black'
                wedge_cmap.set_bad("black", alpha=1.)  
                wedge_cmap.set_under("black")
            
            if titles is not None:
                wedge_title = titles[wedge_idx]

            # read in pixel range 
            cmap_pix_range=projection_data[wedge_idx][1]
            
            # add wedge
            wedge_imgs[wedge_idx], ax  = self.add_wedge(
                axs, wedge_idx, projection_data[wedge_idx][0], 
                wedge_cmap, cmap_pix_range[0], cmap_pix_range[1], 
                beam_radius_deg, wedge_diameter_deg, 
                rmin, rmax, title=wedge_title,
                img_zorder=-1, 
                **img_kwargs
                )
        
        # add axes to the outer edge of the whole beam 
        self.add_beam_axes(axs, redshift_axis_ticks, redshift_axis_minor_ticks, beam_radius_deg, 
            beam_ang_offset=beam_ang_offset_deg, 
            rmin=rmin, rmax=rmax, 
            **axes_kwargs
            )
        
        if filename is not None:
            plt.savefig(f"{filename}", dpi=300, bbox_inches='tight')

            plt.close()
            return wedge_imgs
        if return_ax:
            return axs, wedge_imgs
        elif return_fig:
            return fig, wedge_imgs

    def __define_wedge_params(self, wedge_kwargs=None):
        
        self.wedge_kwargs={
            "alpha":1.0,
            "lw":0.8,
            "ls":"-",
            "edgecolor":"k",
            "facecolor":"none",
            "path_effects":None,
            "zorder":10,
        }
        
        self.wedge_kwargs.update(**(wedge_kwargs or {}))
        # if colour is given, assume its the edge colour and update
        if "color" in self.wedge_kwargs:
            specified_colour = self.wedge_kwargs["color"]
            self.wedge_kwargs.pop("color")
            self.wedge_kwargs["edgecolor"]=specified_colour

    def __define_title_params(self, title_kwargs=None):
        
        self.title_kwargs={
            "color":"white",
            "alpha":1.0,
            "fontsize":10,
            "va":"center",
            "ha":"left",
            "path_effects":[path_effects.withStroke(linewidth=1., foreground="black"), path_effects.Normal()],
            "rotation_mode":"anchor",
        }
        
        self.title_kwargs.update(**(title_kwargs or {}))
        if "rotation" in self.title_kwargs:
            self.title_kwargs.pop("rotation")
        if "zorder" in self.title_kwargs:
            self.title_kwargs.pop("zorder")

    def add_wedge(self, ax, wedge_idx, data_2D, cmap, pix_min, pix_max, beam_max_ang_radius_deg, wedge_ang_diameter_deg, rmin, rmax, 
        title=None, img_zorder=10, wedge_kwargs=None, title_kwargs=None):
        """
        Add each smaller beam or wedge onto the plot. Returns the upadted projected image. 

        :param  wedge_idx:  order that the wedge is added to the plot. 
        :type   wedge_idx:  int
        :param  data_2D:    2D histogram to plot 
        :type   data_2D:    
        :param cmap:  colour map
        :type  cmap:  str or matplotlib colour map type object
        :param  pix_min:    minimum pixel value shown 
        :type   pix_min:    float
        :param  pix_max:    maximum pixel value shown 
        :type   pix_max:    float
        :param  beam_max_ang_radius_deg:    maximum angular radius [deg] of the beam (slice) as a whole
        :type   beam_max_ang_radius_deg:    float
        :param  wedge_ang_diameter_deg:    angular diameter [deg] of each wedge that the beam is split into
        :type   wedge_ang_diameter_deg:    float
        :param  rmin:   minimum comoving distance
        :type   rmin:   float
        :param  rmax:   maximum comoving distance
        :type   rmax:   float
        """

        self.__define_wedge_params(wedge_kwargs)
        self.__define_title_params(title_kwargs)
        
        theta0 = np.deg2rad(beam_max_ang_radius_deg - np.abs(wedge_idx*wedge_ang_diameter_deg))
        theta1 = np.deg2rad(beam_max_ang_radius_deg - np.abs((1+wedge_idx)*wedge_ang_diameter_deg))

        print(f"\twedge {wedge_idx}, theta 0 -> theta 1 = {np.rad2deg(theta0)} ->  {np.rad2deg(theta1)}")

        r = np.linspace(0, rmax, 200)

        # strait lines of wedge
        xi_left  = r * np.cos(theta0)
        yi_left  = r * np.sin(theta0)
        xi_right = r * np.cos(theta1)
        yi_right = r * np.sin(theta1)

        # outer arc coords
        theta_arc = np.linspace(theta0, theta1, 720)
        xi_arc = rmax * np.cos(theta_arc)
        yi_arc = rmax * np.sin(theta_arc)
        # inner arc coords
        xi_inner = rmin * np.cos(theta_arc)
        yi_inner = rmin * np.sin(theta_arc)


        # lines to go around the edge of the wedge
        outer_arc=np.c_[
            xi_arc, yi_arc
        ]
        inner_arc=np.c_[
            xi_inner, yi_inner
        ]
        left_edge = np.c_[
            np.linspace(rmin*np.cos(theta0), rmax*np.cos(theta0), 50),
            np.linspace(rmin*np.sin(theta0), rmax*np.sin(theta0), 50),
        ]
        right_edge = np.c_[
            np.linspace(rmax*np.cos(theta1), rmin*np.cos(theta1), 50),
            np.linspace(rmax*np.sin(theta1), rmin*np.sin(theta1), 50),
        ]


        # draw outline of the wedge
        verts = np.vstack([
            np.c_[xi_inner, yi_inner],              # inner arc
            np.c_[xi_arc, yi_arc][::-1],            # outer arc
        ])

        # plot img of beam wedge 
        img = ax.imshow(
            data_2D.T, 
            norm=col.LogNorm(vmin=pix_min, vmax=pix_max), cmap=cmap, 
            origin="lower", extent=self.axes_extent, zorder=img_zorder,
            interpolation='quadric',
        )
        
        # add wedge to image
        wedge = Polygon(verts, closed=True, **self.wedge_kwargs)
        ax.add_patch(wedge)
        
        # clip to wedge
        img.set_clip_path(wedge)

        if title is not None:
            theta_mid = 0.5 * (theta0 + theta1)

            # Position text on the inner arc
            r_text = rmin

            x = r_text * np.cos(theta_mid)
            y = r_text * np.sin(theta_mid)

            ax.text(
                x,
                y,
                title,
                **self.title_kwargs,
                rotation = np.rad2deg(theta_mid),
                zorder=self.wedge_kwargs["zorder"] + 1,
            )

        return img, ax
        
    def __define_tick_params(self, major_tick_kwargs=None, minor_tick_kwargs=None, tick_label_kwargs=None):
        self.major_tick_kwargs={
            "color":"black",
            "lw":0.8,
            "alpha":1,
        }
        self.major_tick_kwargs.update(**(major_tick_kwargs or {}))
        self.minor_tick_kwargs={
            "color":self.major_tick_kwargs["color"],
            "lw":self.major_tick_kwargs["lw"]*0.55,
            "alpha":self.major_tick_kwargs["alpha"],
        }
        self.minor_tick_kwargs.update(**(minor_tick_kwargs or {}))
        self.tick_label_kwargs={
            "color":"black",
            "alpha":1,
            "fontsize":8,
            "va":"center" , 
            "ha":"right", 
            #"ha":"right",
            'rotation':0,
            'rotation_mode':"anchor",
        }
        self.tick_label_kwargs.update(**(tick_label_kwargs or {}))
    
    def __define_label_params(self, axes_label_kwargs=None):
        self.axes_label_kwargs={
            "color":"black",
            "fontsize":10,
            "weight":"bold",
            "ha":"center",
            "va":"center",
            "rotation_mode":"anchor",
            #"rotation":0,
        }
        
        self.axes_label_kwargs.update(**(axes_label_kwargs or {}))

    def __define_grid_params(self, grid_line_kwargs=None):
        self.grid_line_kwargs={
            "color":"silver",
            "lw":0.8,
            "alpha":1.0,
            'ls':'--'
        }
        
        self.grid_line_kwargs.update(**(grid_line_kwargs or {}))


    def add_beam_axes(self, ax, redshift_major_ticks, redshift_minor_ticks, beam_radius_deg, 
        beam_ang_offset=0., 
        rmin=None, rmax=None,  
        #comoving_distance_major_ticks=np.linspace(0.2, 10, 50), comoving_distance_minor_ticks=np.linspace(0.1, 9.9, 50), comoving_distance_ticks_order_of_mag=None,
        comoving_distance_major_ticks=None, comoving_distance_minor_ticks=None,
        dtheta_major_ticks_deg=5, dtheta_minor_ticks_deg=1, theta_ticks_abs=True, major_tick_length=3.5, minor_tick_length=None,
        redshift_label_offset=(0,0,0), comoving_label_offset=(0,0,0), tick_label_offset=(0,0,0,0),
        comoving_dist_axes_label=r"$\mathrm{comoving~distance}~[{\mathrm{Mpc}}]$", redshift_axes_label=r"$\mathrm{redshift},~z$",
        overlay_grid=(True, True, True), wedge_titles=[],
        major_tick_kwargs=None,minor_tick_kwargs=None,tick_label_kwargs=None, axes_label_kwargs=None, grid_line_kwargs=None,
        ):
        """
        Add axes, labels and ticks to the split beam plot.

        :param  redshift_major_ticks:  location of the major ticks on the redshift axes 
        :type   redshift_major_ticks:  np.ndarray
        :param  redshift_minor_ticks:  location of the minor ticks on the redshift axes 
        :type   redshift_minor_ticks:  np.ndarray
        :param  beam_radius_deg:    maximum angular radius [deg] of the beam (slice) as a whole
        :type   beam_radius_deg:    float
        :param  beam_ang_offset:    decrease the maximum angular radius [deg] shown
        :type   beam_ang_offset:    float
        :param  rmin:   minimum comoving distance [Mpc] of the plot
        :type   rmin:   float
        :param  rmax:   maximum comoving distance [Mpc] on the plot
        :type   rmax:   float
        :param  comoving_distance_major_ticks:  location of the major ticks on the comoving radius axes 
        :type   comoving_distance_major_ticks:  np.ndarray
        :param  comoving_distance_minor_ticks:  location of the minor ticks on the comoving radius axes
        :type   comoving_distance_minor_ticks:  np.ndarray
        :param  dtheta_major_ticks_deg:  spacing [deg] between the major ticks on the angular radius axes
        :type   dtheta_major_ticks_deg:  np.ndarray
        :param  dtheta_minor_ticks_deg:  spacing [deg] between the minor ticks on the angular radius axes
        :type   dtheta_minor_ticks_deg:  np.ndarray
        :param  major_tick_length:  length of major ticks
        :type   major_tick_length:  float
        :param  minor_tick_length:  length of minor ticks
        :type   minor_tick_length:  float
        :param  comoving_dist_axes_label:  label placed on the comoving radius axes
        :type   comoving_dist_axes_label:  str
        :param  redshift_axes_label:  label placed on the redshift axes
        :type   redshift_axes_label:  str
        :param  redshift_label_offset:  additional offset (in coordinate space) added to the redshift axes 
                                            label along the x,y axes & its rotation  (x,y,theta).
        :type   redshift_label_offset:  tuple
        :param  comoving_label_offset:  additional offset (in coordinate space) added to the comoving radius axes 
                                            label along the x,y axes & its rotation  (x,y,theta).
        :type   comoving_label_offset:  tuple
        :param  tick_label_offset:  additional offset (in coordinate space) added to the tick labels on the redshift 
                                        and comoving radius axes (x_redshift,y_redshift, x_radius,y_radius).
        :type   tick_label_offset:  tuple
        :param  overlay_grid:   If true overlay a grid line at each major tick from the given axes (redshift, comoving radius, angular radius)
        :type   overlay_grid:   tuple    (boolean, boolean, boolean)
        :param  major_tick_kwargs:    keyword arguments to control the major tick lines 
        :type   major_tick_kwargs:    dict
        :param  minor_tick_kwargs:    keyword arguments to control the minor tick lines 
        :type   minor_tick_kwargs:    dict
        :param  tick_label_kwargs:    keyword arguments to control the text of the labels on the major ticks
        :type   minor_tick_kwargs:    dict
        :param  axes_label_kwargs:    keyword arguments to control the text of the axes labels
        :type   axes_label_kwargs:    dict
        :param  grid_line_kwargs:    keyword arguments to control the grid lines
        :type   grid_line_kwargs:    dict
        """
        ax.figure.canvas.draw() # require ax to be drawn so transformation from pixels to axes coords are stable

        # define basic properties of ticks and labels 
        self.__define_tick_params(major_tick_kwargs, minor_tick_kwargs, tick_label_kwargs)
        self.__define_label_params(axes_label_kwargs)
        self.__define_grid_params(axes_label_kwargs)

        # tick properties
        tick_len = major_tick_length  #12
        minor_tick_len=tick_len*0.55
        tick_fontsize=self.tick_label_kwargs["fontsize"] #needed for placing labels
        
        # handle rotation of tick labels 
        if 'rotation' in self.tick_label_kwargs:
            if self.tick_label_kwargs['rotation']=="auto":
                auto_rot_tick_label=True
                self.tick_label_kwargs.pop('rotation')
            else:
                auto_rot_tick_label=False
        
        # fontsize for axes labels and ticks
        label_fontsize=self.axes_label_kwargs["fontsize"] #needed for placing labels

        # zorder
        axes_top_level=20
        axes_low_level=10
        axes_overlay_level=30

        # define wedge angles
        theta_max = np.deg2rad(beam_radius_deg)  # max theta from 0

        # comoving distances from redshifts
        comoving_distances=self.cosmology.comoving_distance(redshift_major_ticks).to_value("Mpc")
        comoving_tick_range = [np.min(comoving_distances), np.max(comoving_distances)]

        if redshift_minor_ticks is not None:
            minor_comoving_distances=self.cosmology.comoving_distance(redshift_minor_ticks).to_value("Mpc")
            
            if np.min(minor_comoving_distances) < comoving_tick_range[0]:
                comoving_tick_range[0] = np.min(minor_comoving_distances)
            if np.max(minor_comoving_distances) > comoving_tick_range[1]:
                comoving_tick_range[1] = np.max(minor_comoving_distances)

        #image coordinate max and min radius 
        if rmax is None:
            rmax =comoving_distances[-1]+1
        ## set inner arc radius
        if rmin is  None:
            rmin=0

        r = np.linspace(0, rmax, 200)
        
        ##### x,y coords for whole beam & arc #####

        #edges
        x_left  = r * np.cos(theta_max)
        y_left  = r * np.sin(theta_max)
        x_right = r * np.cos(-theta_max)
        y_right = r * np.sin(-theta_max)

        # outer arc
        theta_arc = np.linspace(-theta_max, theta_max, 720)
        x_arc = rmax * np.cos(theta_arc)
        y_arc = rmax * np.sin(theta_arc)

        # set axes locations
        x_mid_l = (0.5 * (rmax-rmin) + rmin) * np.cos(theta_max)
        x_mid_r = (0.5 * (rmax-rmin) + rmin) * np.cos(-theta_max)
        y_mid_l = (0.5 * (rmax-rmin) + rmin) * np.sin(theta_max)
        y_mid_r = (0.5 * (rmax-rmin) + rmin) * np.sin(-theta_max)


        # add auto rotation to label and then remove after it is written 
        if auto_rot_tick_label:
            label_rotation =np.rad2deg(theta_max)
            if label_rotation>90:
                label_rotation -= 180
            elif label_rotation< -90:
                label_rotation += 180
            self.tick_label_kwargs['rotation']=label_rotation
        
        offset_direction=-1
        
        for ii, rr in enumerate(comoving_distances[:-1]):
            x0 = rr * np.cos(theta_max)
            y0 = rr * np.sin(theta_max)

            # points in deraction of the tick
            dx = -np.sin(theta_max)
            dy = np.cos(theta_max)

            tx_in, ty_in = self.offset_point_from_tick(ax, x0, y0, dx, dy, -tick_len)
            tx_out, ty_out = self.offset_point_from_tick(ax, x0, y0, dx, dy, tick_len)

            # plot tick
            ax.plot(
                [tx_in, tx_out],
                [ty_in, ty_out],
                **self.major_tick_kwargs,
                zorder=axes_top_level
            )
            
            # redshift tick labels
            lx, ly = self.offset_point_from_tick(ax, x0, y0, dx, dy, -1*offset_direction * (0.5*tick_fontsize + 1.*tick_len))
        
            ax.text(
                lx+tick_label_offset[0], ly+tick_label_offset[0],
                f"{redshift_major_ticks[ii]:.2f}", 
                **self.tick_label_kwargs,
                zorder=axes_top_level)
            
            # draw grid  
            if overlay_grid[0]:
                x,y = self.arc_xy(rr, beam_radius_deg)
                ax.plot(x, y,**self.grid_line_kwargs,zorder=axes_low_level)
                
        if auto_rot_tick_label:
            self.tick_label_kwargs.pop('rotation')

        if redshift_minor_ticks is not None:

            for ii, rr in enumerate(minor_comoving_distances[:-1]):
                x0 = rr * np.cos(theta_max)
                y0 = rr * np.sin(theta_max)

                # small perpendicular tick
                dx = -np.sin(theta_max)
                dy = np.cos(theta_max)


                tx_in, ty_in = self.offset_point_from_tick(ax, x0, y0, dx, dy, -minor_tick_len)
                tx_out, ty_out = self.offset_point_from_tick(ax, x0, y0, dx, dy, minor_tick_len)

                ax.plot(
                    [tx_in, tx_out],
                    [ty_in, ty_out],
                    **self.minor_tick_kwargs,
                    zorder=axes_top_level
                )

        # redshift axes label 
        p1 = (rmin * np.cos(theta_max), rmin * np.sin(theta_max)) 
        p2 = (rmax * np.cos(theta_max), rmax * np.sin(theta_max)) 
        px_offset = (label_fontsize/2  +  3*tick_fontsize + 2)
        label_x, label_y = self.labels_loc(ax, p1, p2, offset=px_offset, dpi=None, tick_length=1.75*tick_len)

        label_rotation =np.rad2deg(theta_max)
        if label_rotation>90:
            label_rotation -= 180
        elif label_rotation< -90:
            label_rotation += 180
        
        ax.text(
                label_x+redshift_label_offset[0], label_y+redshift_label_offset[1],
                redshift_axes_label,
                rotation=label_rotation + redshift_label_offset[2],
                **self.axes_label_kwargs,
            )


        # include comoving distance axes and ticks 

        if comoving_distance_major_ticks is None:
            # order of magnitude of most distant comoving tick 
            order_of_mag = orderOfMagnitude(comoving_tick_range[-1])
            major_tick_spacing = round_down_10(self.minor_tick_spacer((comoving_tick_range[-1]-comoving_tick_range[0]), 10, 2))
            if major_tick_spacing>0:
                start_val = int(comoving_tick_range[0] / (1*10**(orderOfMagnitude(comoving_tick_range[0]))))
                comoving_distance_major_ticks = np.arange(0, 2*10**(order_of_mag+1), major_tick_spacing)
            else:
                comoving_distance_major_ticks = None
                
        if comoving_distance_minor_ticks is None:
            if comoving_distance_major_ticks is None:
                major_tick_spacing = comoving_tick_range[-1]-comoving_tick_range[0]
            else:
                major_tick_spacing = comoving_distance_major_ticks[1]-comoving_distance_major_ticks[0]
            order_of_mag = orderOfMagnitude(comoving_tick_range[-1])
            minor_tick_spacing = self.minor_tick_spacer(major_tick_spacing, 5, 1)
            comoving_distance_minor_ticks = np.arange(0, 2*10**(order_of_mag+1), minor_tick_spacing)
        
        # clip tick locations 
        if comoving_distance_major_ticks is not None:
            comoving_distance_major_ticks = comoving_distance_major_ticks[(comoving_distance_major_ticks>=comoving_tick_range[0]) & (comoving_distance_major_ticks<=comoving_tick_range[-1])]
        if comoving_distance_minor_ticks is not None:
            comoving_distance_minor_ticks = comoving_distance_minor_ticks[(comoving_distance_minor_ticks>=comoving_tick_range[0]) & (comoving_distance_minor_ticks<=comoving_tick_range[-1])]


        # add auto rotation to label and then remove after it is written 
        if auto_rot_tick_label:
            label_rotation =np.rad2deg(-theta_max)
            if label_rotation>90:
                label_rotation += 180
            elif label_rotation< -90:
                label_rotation -= 180
            self.tick_label_kwargs['rotation']=label_rotation


        offset_direction=-1

        if comoving_distance_major_ticks is not None:
            for ii, rr in enumerate(comoving_distance_major_ticks):

                x0 = rr * np.cos(-theta_max)
                y0 = rr * np.sin(-theta_max)

                # small perpendicular tick
                dx = -np.sin(-theta_max)
                dy = np.cos(-theta_max)

                tx_in, ty_in = self.offset_point_from_tick(ax, x0, y0, dx, dy, -1 * offset_direction * tick_len)
                tx_out, ty_out = self.offset_point_from_tick(ax, x0, y0, dx, dy, offset_direction * tick_len)

                ax.plot(
                    [tx_in, tx_out],
                    [ty_in, ty_out],
                    **self.major_tick_kwargs,
                    zorder=axes_top_level
                )
                

                lx, ly = self.offset_point_from_tick(ax, x0, y0, dx, dy, offset_direction * (0.5*tick_fontsize + 1.*tick_len))
                ax.text(
                    lx+tick_label_offset[2], ly+tick_label_offset[3],
                    f"{math.floor(rr):d}", 
                    **self.tick_label_kwargs,
                    zorder=axes_top_level)
                
                # draw grid  
                if overlay_grid[1]:
                    x,y = self.arc_xy(rr, beam_radius_deg)
                    ax.plot(x, y,**self.grid_line_kwargs,zorder=axes_low_level)


        if comoving_distance_minor_ticks is not None:
            for ii, rr in enumerate(comoving_distance_minor_ticks):
                x0 = rr * np.cos(-theta_max)
                y0 = rr * np.sin(-theta_max)

                # small perpendicular tick
                nx = -np.sin(-theta_max)
                ny = np.cos(-theta_max)

                tx_in, ty_in = self.offset_point_from_tick(ax, x0, y0, nx, ny, -minor_tick_len)
                tx_out, ty_out = self.offset_point_from_tick(ax, x0, y0, nx, ny, minor_tick_len)

                ax.plot(
                    [tx_in, tx_out],
                    [ty_in, ty_out],
                    **self.minor_tick_kwargs,
                    zorder=axes_top_level
                )
        if auto_rot_tick_label:
            self.tick_label_kwargs.pop('rotation')
        
        if comoving_distance_major_ticks is not None: # do not show a label if there are no major ticks
            # comoving distance axes labels
            p1 = (rmin * np.cos(-theta_max), rmin * np.sin(-theta_max)) 
            p2 = (rmax * np.cos(-theta_max), rmax * np.sin(-theta_max))
            px_offset = (label_fontsize/2  +  3*tick_fontsize + 2) # offset > fontsize/2 + tick fontsize + extra space
            label_x, label_y = self.labels_loc(ax, p1, p2, offset=-px_offset, dpi=None, tick_length=-1.75*tick_len)

            # rotate to keep inline with 
            label_rotation =np.rad2deg(-theta_max)
            if label_rotation>90:
                label_rotation += 180
            elif label_rotation< -90:
                label_rotation -= 180

            ax.text(
                    label_x+comoving_label_offset[0], label_y+comoving_label_offset[1],
                    comoving_dist_axes_label,
                    rotation=label_rotation+comoving_label_offset[2],
                    **self.axes_label_kwargs,
                )

        # add arc annotation for angle

        tick_str_format="{tick_val:.1f}°"

        if dtheta_major_ticks_deg is None:
            dtheta=None
        else:
            dtheta=dtheta_major_ticks_deg

        if (dtheta_minor_ticks_deg is None):
            dtheta_minor = None
        elif dtheta_minor_ticks_deg == "auto":
            dtheta_minor = self.minor_tick_spacer
        else:
            dtheta_minor=dtheta_minor_ticks_deg

        # automatically adjust cadance of ticks if angle is very large
        if (beam_radius_deg > 45) and (dtheta<=5):
            #print(beam_radius_deg*2, beam_radius_deg%30)
            if (beam_radius_deg > 90) and ((beam_radius_deg*2)%30==0):
                dtheta=30 # always prefer 30degree over other seperations 
            else:
                dtheta_test = self.minor_tick_spacer(beam_radius_deg*2, 10, 5)
                if dtheta_test < 10:
                    dtheta=10
                else:
                    dtheta = round_down_10(dtheta_test)

            dtheta_minor = self.minor_tick_spacer(dtheta)

        # do not include minor ticks if they are greater than major ticks in distance
        if dtheta_minor >= dtheta:
            dtheta_minor=None

        #angle_ticks = np.arange(-1*(beam_radius_deg+beam_ang_offset/2), (beam_radius_deg+beam_ang_offset/2)+dtheta, dtheta)
        if dtheta is None:
            angle_ticks =None
        else:
            radius_ticks = np.arange(0, (beam_radius_deg+beam_ang_offset/2)+dtheta, dtheta)
            radius_ticks = radius_ticks[radius_ticks<=beam_radius_deg]
            angle_ticks  = np.concatenate((-1*radius_ticks[::-1], radius_ticks[1:]))

        if dtheta_minor is None:
            angle_minor_ticks=None
        else:
            radius_minor_ticks = np.arange(0+dtheta_minor, (beam_radius_deg+beam_ang_offset/2)+dtheta_minor, dtheta_minor)
            radius_minor_ticks = radius_minor_ticks[radius_minor_ticks<=beam_radius_deg]
            angle_minor_ticks  = np.concatenate((-1*radius_minor_ticks[::-1], radius_minor_ticks))

        if angle_ticks is not None:
            self.tick_label_kwargs["va"]="center" # need to update for ang ticks
            self.tick_label_kwargs["ha"]="center" # need to update for ang ticks
            if 'rotation' in self.tick_label_kwargs:
                self.tick_label_kwargs.pop('rotation')

            for a in angle_ticks:
                if np.abs(a) > beam_radius_deg:
                    continue 
                elif np.abs(a) == beam_radius_deg:
                    a= a/np.abs(a) * beam_radius_deg

                th = np.deg2rad(a)

                # tick on arc
                x0 = rmax * np.cos(th)
                y0 = rmax * np.sin(th)

                nx, ny = -np.cos(th), -np.sin(th)

                tx_in, ty_in = self.offset_point_from_tick(ax, x0, y0, -nx, -ny, -tick_len)
                tx_out, ty_out = self.offset_point_from_tick(ax, x0, y0, -nx, -ny, tick_len)
                ax.plot(
                    [tx_in, tx_out],
                    [ty_in, ty_out],
                    **self.major_tick_kwargs,
                    zorder=axes_top_level
                )
                if overlay_grid[2] and np.abs(a)+1 < beam_radius_deg:

                    ax.plot(
                        [rmin * np.cos(th), tx_in],
                        [rmin * np.sin(th), ty_in],
                        **self.grid_line_kwargs,
                        zorder=axes_low_level
                    )

                # rotation parallel to the tick mark
                rot = a
                #a = a if theta_label_deg else a * np.pi/180
                theta_tick_val = np.abs(a) if theta_ticks_abs else a

                # label slightly outside arc
                label_pad = 1 #1.5 * tick_len
                font_offset =  1.5*tick_fontsize if len(str(theta_tick_val)) <=4 else 1.75*tick_fontsize

                lx, ly = self.offset_point_from_tick(ax, x0, y0, -nx, -ny, font_offset + 1*tick_len)

                ax.text(
                    lx, ly, # + tick_len, 
                    tick_str_format.format(tick_val=theta_tick_val), 
                    rotation=rot, 
                    **self.tick_label_kwargs
                    )

        if angle_minor_ticks is not None:
            for a in angle_minor_ticks:

                if np.abs(a) > beam_radius_deg:
                    continue 
                elif np.abs(a) == beam_radius_deg:
                    a= a/np.abs(a) * beam_radius_deg

                th = np.deg2rad(a)

                # tick on arc
                x0 = rmax * np.cos(th)
                y0 = rmax * np.sin(th)

                # inward normal direction (radial inward)
                nx, ny = -np.cos(th), -np.sin(th)

                tx_in, ty_in = self.offset_point_from_tick(ax, x0, y0, -nx, -ny, -minor_tick_len)
                tx_out, ty_out = self.offset_point_from_tick(ax, x0, y0, -nx, -ny, minor_tick_len)

                ax.plot(
                    [tx_in, tx_out],
                    [ty_in, ty_out],
                    **self.minor_tick_kwargs,
                    zorder=axes_top_level
                )


    @staticmethod
    def arc_xy(comoving_dist, theta_degree, n=360):
        """
        Get point on the arc of the beam
        """
        #if theta_degree is None:
        if theta_degree is not None:
            theta_rad = (np.deg2rad(-theta_degree), np.deg2rad(theta_degree))
        elif theta_degree is None:
            theta_rad = (0, 2*np.pi)
        arc_rad = np.linspace(theta_rad[0], theta_rad[1], n)
        a = comoving_dist * np.cos(arc_rad)
        b = comoving_dist * np.sin(arc_rad)
        return a, b

    @staticmethod
    def minor_tick_spacer(major_spacing, max_numb_of_minor_ticks=5, min_numb_of_minor_ticks=2):
        """
        Return minor tick spacing for preferred number of minor ticks between major ticks.
        """
        n_minor_divisions = np.arange(min_numb_of_minor_ticks+1, max_numb_of_minor_ticks+2, 1)[::-1]
        for divisions in n_minor_divisions:  # 5,4,3,2 minor ticks
            minor = major_spacing / divisions
            exponent = np.floor(np.log10(abs(minor)))
            mantissa = minor / 10**exponent
            if np.isclose(mantissa, (1, 2, 2.5, 5, 10, 30)).any():
                return minor
        # if all else fails return 2 minor ticks
        return major_spacing / 3

    @staticmethod
    def labels_loc(ax, p1, p2, offset=12, dpi=None, tick_length=1):
        """
        Determine the position and rotation angle of the axes labels
        based on 2 points along the axes
        """

        # Offset in display coordinates (points -> pixels)
        if dpi is None:
            dpi = ax.figure.dpi
            
        offset_px = (tick_length+offset) * dpi / 72

        # Transform to display coordinates
        #p1 = (rmin * np.cos(-theta_max), rmin * np.sin(-theta_max))
        #p2 = (rmax * np.cos(-theta_max), rmax * np.sin(-theta_max))
        #p1 = (x_right[0], y_right[0]) #np.linspace(rmin, rmax, 200) * np.cos(-theta_max)
        #p2 = (x_right[-1], y_right[-1]) #np.linspace(rmin, rmax, 200) * np.sin(-theta_max)

        # points in display coords
        t = ax.transData.transform
        x1, y1 = t(p1)
        x2, y2 = t(p2)
        
        # change in positon 
        dx = x2 - x1
        dy = y2 - y1
        
        # norm
        length = np.hypot(dx, dy)
        nx = -dy / length
        ny = dx / length

        x_disp = (x1+x2) / 2 + nx * offset_px
        y_disp = (y1+y2) / 2 + ny * offset_px

        # Back to data coordinates
        x_data, y_data = ax.transData.inverted().transform((x_disp, y_disp))

        return x_data, y_data

    @staticmethod
    def offset_point_from_tick(ax, x, y, ux, uy, offset):
        """
        Use display coordinates to move tick label away 
        from the tick by shifting it the direction (ux,uy)
        """

        # Tick endpoint in display coordinates
        x_disp, y_disp = ax.transData.transform((x, y))

        # radial direction in display coordinates
        x2_disp, y2_disp = ax.transData.transform((x + ux, y + uy))

        dx = x2_disp - x_disp
        dy = y2_disp - y_disp
        #print(f"\t norm: {np.hypot(dx, dy)}")
        norm = np.hypot(dx, dy)
        dx /= norm
        dy /= norm

        # points -> pixels
        offset_pixels = offset * ax.figure.dpi / 72

        # apply offset
        x_new_disp = x_disp + dx * offset_pixels
        y_new_disp = y_disp + dy * offset_pixels

        # back to data coordinates
        return ax.transData.inverted().transform(
            (x_new_disp, y_new_disp)
        )

    @staticmethod
    def points_to_data(ax, x, y, dx, dy, offset):
        """
        Use display coordinates to move points on a plot
        """
        # norm
        L = np.hypot(dx, dy)
        dx /= L
        dy /= L

        # transform points -> display coords
        x0, y0 = ax.transData.transform((x, y))

        # determine offset with pixel per sqr inch 
        offset_px = offset * ax.figure.dpi / 72 # 1 point = dpi / 72 pixels

        # change position in display coords
        x1 = x0 + offset_px * dx
        y1 = y0 + offset_px * dy

        # display coords -> data coords
        xd, yd = ax.transData.inverted().transform((x1, y1))

        return xd - x, yd - y



def return_circle_mollweide(lon0, lat0, radius_deg, n=500):
    """
    Draw a true spherical circle on a Matplotlib Mollweide axis using ax.plot.
    """
    # convert to radians
    lon0 = float(np.squeeze(lon0))
    lat0 = float(np.squeeze(lat0))
    lon0 = np.deg2rad(lon0)
    lat0 = np.deg2rad(lat0)
    r = np.deg2rad(radius_deg)
    # center vector
    c = np.array([
        np.cos(lat0) * np.cos(lon0),
        np.cos(lat0) * np.sin(lon0),
        np.sin(lat0)
    ])
    # orthonormal basis on tangent plane
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(c, z):
        z = np.array([1.0, 0.0, 0.0])
    u = np.cross(z, c)
    u /= np.linalg.norm(u)
    v = np.cross(c, u)
    t = np.linspace(0, 2*np.pi, n)
    # circle on sphere
    pts = (
        np.cos(r) * c[:, None] +
        np.sin(r) * (np.outer(u, np.cos(t)) + np.outer(v, np.sin(t)))
    ).T
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    lon = np.arctan2(y, x)
    lat = np.arcsin(z)
    # wrap longitude for Mollweide continuity
    lon = (lon + np.pi) % (2*np.pi) - np.pi
    return lon, lat

class add_to_mollweide:
    """
    Class of functions to add more information too mollweide projections of the lightcones.
    !!! OVERLAID COORDINATES LOOSE PRECISION FOR LATITUDE!=0 !!! 
    On going work to fix the coordinate transforms between matplotlib and healpy newvisufunc, therefore 
    precision of the overlaid matplotlib coorindtes onto a mollweide projection may loose precision at non-zero 
    latitudes.

    """
    def __init__(self, add_to_ax=None, add_to_fig=None):
        
        self.map_ax=add_to_ax
        self.map_fig=add_to_fig
        


    @staticmethod
    def overlay_new_mollweide_graticule(
            ax=None, 
            latitude_deg=[-60, -30, 0, 30, 60], 
            longitude_deg=[-120, -60, 0, 60, 120], 
            range_y=(-90, 90), 
            range_x=(-180, 180),
            label_x_offset=0.02, 
            label_y_offset=0.,
            line_patheffects=None,
            label_patheffects=None,
            npoints=360,
            zorders=(None, 10),
            **kwargs):
        """
        Draw latitude and longitude lines.

        :param  ax:                             matplotlib.pyplot object for axes of the plot, if None then construct a new axes ontop of existing figures
        :param  latitude_deg:                   array of latitude coordinates to draw lines along [degrees]
        :param  longitude_deg:                  array of longitude coordinates to draw lines along [degrees]
        :param  range_y:                        start and end points of overlaid lines in terms of longitude [degree], 
        :param  range_x:                        start and end points of overlaid lines in terms of latitude [degree], 
        :param  npoints:                        number of points sampled when drawing lines. M
        :param  zorders:                        tuple (lines, labels), the zorder of the overlaid graticule lines and labels.
        :param  label_patheffects:              path effects added to the overlaid text, default = None 
        :param  line_patheffects:               path effects added to the overlaid lines, default = None 
        """

        # define dictionaries for styling the overlaid graticules
        graticule_dict={
            'lon_linecolour':'k', 
            'lon_linewidth':plt.rcParams["lines.linewidth"],
            'lon_linestyle':plt.rcParams['lines.linestyle'],
            'lon_labelcolour':'k', 
            'lon_labelsize':mpl.rcParams['ytick.labelsize'],
            #
            'lat_linecolour':'k', 
            'lat_linewidth':plt.rcParams["lines.linewidth"],
            'lat_linestyle':plt.rcParams['lines.linestyle'],
            'lat_labelcolour':'k', 
            'lat_labelsize':mpl.rcParams['xtick.labelsize'],
            #
            'centre_linewidth':plt.rcParams["lines.linewidth"],
            'centre_linecolour':None,
            'centre_linestyle':plt.rcParams["lines.linestyle"],
            'dash_capstyle':plt.rcParams["lines.dash_capstyle"],
            'solid_capstyle':plt.rcParams["lines.solid_capstyle"],
            'line_alpha':1.,
            'label_alpha':1.,
            } 

        # update undefined niche kwargs from simple kwargs
        general_kwargs = ["linestyle", "linewidth", "linecolour", "labelcolour", "labelsize"]
        for general_key in general_kwargs:
            for graticule_ax in ["lon", "lat"]:
                graticule_key = graticule_ax+"_"+general_key
                if (graticule_key not in kwargs) and (general_key in kwargs):
                    kwargs[graticule_key] = kwargs[general_key]

        ## update centre kwargs from simple kwargs
        if ("centre_linestyle" not in kwargs) and ("linestyle" in kwargs):
            kwargs["centre_linestyle"] = kwargs["linestyle"]

        if ("centre_linewidth" not in kwargs) and ("linewidth" in kwargs):
            kwargs["centre_linewidth"] = kwargs["linewidth"]

        if 'line_patheffects' in kwargs:
            grat_path_list = kwargs['line_patheffects']
        if 'label_patheffects' in kwargs:
            label_path_list = kwargs['label_patheffects']

        graticule_dict.update(kwargs)  # only provided keys overwrite defaults

        ## overlay new graticule lines
        if ax is None:
            ax_main = plt.gca()
        else:
            ax_main = ax

        # plot central lines
        if graticule_dict['centre_linecolour']  is None:
           centre_col = graticule_dict['lon_linecolour']

        ax_main.plot(np.zeros(2), np.deg2rad([range_x[0], range_x[1]]), 
            color=centre_col, 
            lw=graticule_dict['centre_linewidth'], 
            linestyle=graticule_dict["centre_linestyle"],
            dash_capstyle=graticule_dict['dash_capstyle'],
            #capstyle=graticule_dict['solid_capstyle'],
            alpha=graticule_dict['line_alpha'],
            path_effects=line_patheffects,
            zorder=zorders[0]
            )

        x=np.linspace(np.deg2rad(range_x[0]), np.deg2rad(range_x[1]), npoints)
        for lat in latitude_deg:
            phi = np.radians(lat)
            #y = np.full_like(x, lat_to_mollweide_y(phi))
            __, lat_on_circle = return_circle_mollweide(0, lat, 0)
            y=np.full_like(x, np.max(lat_on_circle))
            ax_main.plot(
                x, y,
                color=graticule_dict['lat_linecolour'],
                lw=graticule_dict['lat_linewidth'],
                linestyle=graticule_dict['lat_linestyle'],
                alpha=graticule_dict['line_alpha'],
                path_effects=line_patheffects,
                dash_capstyle=graticule_dict['dash_capstyle'],
                zorder=zorders[0]
                #capstyle=graticule_dict['solid_capstyle'],

            )

        if graticule_dict['centre_linecolour']  is None:
           centre_col = graticule_dict['lat_linecolour']

        ax_main.plot(np.deg2rad([range_y[0], range_y[1]]), np.zeros(2), 
            color=centre_col, 
            lw=graticule_dict['centre_linewidth'], 
            linestyle=graticule_dict["centre_linestyle"],
            dash_capstyle=graticule_dict['dash_capstyle'],
            #capstyle=graticule_dict['solid_capstyle'],
            alpha=graticule_dict['line_alpha'],
            path_effects=line_patheffects,
            zorder=zorders[0]
            )

        y = np.linspace(np.deg2rad(range_y[0]), np.deg2rad(range_y[1]), npoints)

        for lon in longitude_deg:
            if lon==0:
                continue
            #x = np.radians(lon) * np.ones_like(y)
            lon_on_circle, __ = return_circle_mollweide(lon, 0, 0)
            x=np.max(lon_on_circle)* np.ones_like(y)
            ax_main.plot(
                x, y,
                color=graticule_dict['lon_linecolour'],
                lw=graticule_dict['lon_linewidth'],
                linestyle=graticule_dict['lon_linestyle'],
                alpha=graticule_dict['line_alpha'],
                path_effects=line_patheffects,
                dash_capstyle=graticule_dict['dash_capstyle'],
                zorder=zorders[0]
                #capstyle=graticule_dict['solid_capstyle'],
            )


        # overlay labels
        for lon_deg in longitude_deg:
            if lon_deg!=0:
                lon_label = np.abs(lon_deg) if lon_deg < 0 else 360-lon_deg
                y_offset=label_y_offset
            else:
                lon_label = 0

            lon_on_circle, __ = return_circle_mollweide(lon_deg, 0, 0)
            x=np.max(lon_on_circle)
            ax_main.text(
                #np.radians(lon_deg),
                x,
                0,
                f"{lon_label}"+r"$^{\circ}$",
                color=graticule_dict['lon_labelcolour'],
                ha="center",
                va="center",
                fontsize=graticule_dict['lon_labelsize'],
                path_effects=label_patheffects,
                zorder=zorders[-1]
                #path_effects=[path_effects.withStroke(linewidth=1.2, foreground="black"), path_effects.Normal()],   
            )
        for lat_deg in latitude_deg:
            if lat_deg !=0:
                lat_va= "bottom" if lat_deg>0 else "top"
                x_offset= 0. if lat_deg>0 else 0.02
            else:
                lat_va="center"
                x_offset=0.02

            __, lat_on_circle = return_circle_mollweide(0, lat_deg, 0)
            ax_main.text(
                -np.pi - x_offset,
                #mollweide_lat(np.radians(lat_deg)),
                np.max(lat_on_circle),
                f"{lat_deg}"+r"$^{\circ}$",
                color=graticule_dict['lat_labelcolour'],
                ha="right",
                va=lat_va,
                fontsize=graticule_dict['lat_labelsize'],
                path_effects=label_patheffects,
                zorder=zorders[-1]
                #path_effects=[path_effects.withStroke(linewidth=1.2, foreground="black"), path_effects.Normal()],   
            )



    @staticmethod
    def enforce_colourbar_minor_ticks(ax, scale='log', n=4, tick_length=(None, None), tick_width=(None, None)):
        """
        For Projview function in healpy enforce minot ticks on the colour bar.

        :param  n:              number of minor ticks to include. 
        :type   n:              int
        :param  scale:          the scaling type of the colour bar. 
                                'log', colour bar has norm='log'. 
                                'hist', colour bar has norm='hist', add minor ticks at the location of the minor ticks 
                                    with the non-uniform histogram normalisation. 
                                'transform', place minor ticks at the locations based on the transform 
                                    function of the colourbars axes. Works best for norm='none'.
        :type   scale:          str
        :param  tick_length:    tuple (major, minor) of tick lengths
        :type   tick_length:    tuple (float or int, float or int)
        :param  tick_width:     tuple (major, minor) of tick widths
        :type   tick_width:     tuple (float or int, float or int)
        """


        def add_hist_minor_ticks(ax, n):
            mesh = next(c for c in ax.collections if hasattr(c, "norm"))
            norm = mesh.norm
            major = ax.get_xticks()
            # Map major ticks into normalized colorbar coordinates
            u = norm(major)
            minor = []
            for a, b in zip(u[:-1], u[1:]):
                # n minor ticks between each pair
                uu = np.linspace(a, b, n + 2)[1:-1]
                minor.extend(norm.inverse(uu))
            ax.xaxis.set_minor_locator(ticker.FixedLocator(minor))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())

        if scale=='log':
            ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10)) # make major ticks 10^x
            ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2,10)*0.1))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        elif scale=='hist':
            mesh = next(c for c in ax.collections if hasattr(c, "norm"))
            norm = mesh.norm
            major = ax.get_xticks()
            # Map major ticks into normalized colorbar coordinates
            u = norm(major)
            minor = []
            for a, b in zip(u[:-1], u[1:]):
                # n minor ticks
                uu = np.linspace(a, b, n + 2)[1:-1]
                minor.extend(norm.inverse(uu))
            ax.xaxis.set_minor_locator(ticker.FixedLocator(minor))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        elif scale=="transform":
            major = ax.get_xticks()
            tr = ax.xaxis._scale._transform # transform functions
            forward = tr._forward      # transform forward
            inverse = tr._inverse      # transform inverse
            minor = []
            for a, b in zip(major[:-1], major[1:]):
                ta = forward(a)
                tb = forward(b)
                # n minor ticks
                t = np.linspace(ta, tb, n+2)[1:-1]
                minor.extend(inverse(t))
            ax.xaxis.set_minor_locator(ticker.FixedLocator(minor))

        else:
            raise ValueError("scale not recognised, use: 'log','hist' or 'transform'")
        
        def colorbar_orientation(ax):
            bbox = ax.get_position()
            if bbox.width > bbox.height:
                return "horizontal"
            else:
                return "vertical"

        if colorbar_orientation(ax) =="horizontal":
            len0=plt.rcParams["xtick.major.size"] if tick_length[0] is None else tick_length[0]
            len1=plt.rcParams["xtick.minor.size"] if tick_length[1] is None else tick_length[1]
            wd0=plt.rcParams["xtick.major.width"] if tick_width[0] is None else tick_width[0]
            wd1=plt.rcParams["xtick.minor.width"] if tick_width[1] is None else tick_width[1]
            ax.tick_params(axis='x', direction='inout', length=len1, width=wd1, which="minor")
            ax.tick_params(axis='x', direction='inout', length=len0, width=wd0, which="major")
        else:
            len0=plt.rcParams["ytick.major.size"]  if tick_length[0] is None else tick_length[0]
            len1=plt.rcParams["ytick.minor.size"]  if tick_length[1] is None else tick_length[1]
            wd0= plt.rcParams["ytick.major.width"] if tick_width[0] is None else tick_width[0]
            wd1= plt.rcParams["ytick.minor.width"] if tick_width[1] is None else tick_width[1]
            ax.tick_params(axis='y', direction='inout', length=len1, width=wd1, which="minor")
            ax.tick_params(axis='y', direction='inout', length=len0, width=wd0, which="major")


    @staticmethod
    def add_zoom_region(fig, dset,
        pix_min=1e-10, pix_max=1, 
        zoom_vec=np.array([-0.75, 0.54, 0.366]), window_vec=[0.675,-0.4,0.6,0.6],
        xy_npix=(480,480), pix_res=0.5, 
        scale_bar=(0.5, r"$30 ^{\prime}$", "white"),
        norm="log",
        cmap=None,
        window_text=None, 
        text_size=None,
        text_colour="white", 
        text_path_effects=[path_effects.withStroke(linewidth=1., foreground="black"), path_effects.Normal()],
        text_background=False,
        connector_colour="silver",
        connector_points=None,
        window_zorder=(100,200),
        scale_xy_shift=(0,0),
        return_img=False
        ):
        """
        Create a Gnomonic Projection as an inset zoom region. 
        Show a window inset on the larger figure that contains a zoomed in region of the 
        larger map + overlay a corresponding small region on the healpix map showing the 
        footprint of the inset window. 

        :param  fig:                matplotlib figure object to overlay zoomed region on
        :type   fig:                matplotlib.figure.Figure 
        :param  dset:               Healpy Map dataset. 
        :type   dset:               np.ndarray, numpy.ma.MaskedArray or list 
        :param  pix_min, pix_max:   min and max values of pixels in zoomed region
        :type   pix_min, pix_max:   float or int
        :param  zoom_vec:           vector that indicates the centre of the zoomed region on the larger HEALPix map
                                        i.e. on sky vector as given in healpy
        :type   zoom_vec:           np.ndarray or list
        :param  window_vec:         location of the zoomed regions lower left corner on 
                                        the larger plot + the windows width and height [xmin, ymin, width, height]
        :type   window_vec:         np.ndarray or list
        :param  scale_bar:          (angular size [arcmin], label, colour), the angular size of the scale bar 
                                        in arcminutes,  the label shown and the colour of the bar 
        :type   scale_bar:          tuple (float, str, str)
        :param  xy_npix:            (n, n) number of pixels on the x, y axes. 
        :type   xy_npix:            tuple (int, int)
        :param  pix_res:            angular size of pixels [arcmin]
        :type   pix_res:            float or int
        :param  text_size:          fontsize of text (scale and additional text)
        :type   text_size:          float or int
        :param  text_colour:        colour of text overlaid on image
        :type   text_colour:        str or colour map object
        :param  text_path_effects:  path effects added to the text, if None, then no affects applied
        :type   text_path_effects:  list of path effects arguments
        :param  scale_xy_shift:     additional offset added to the location of the scale bar within the inset window, in axes coordates. 
        :type   scale_xy_shift:     typle (float, float)
        :param  connector_colour:   colour of lines connecting the zoomed window to the healpy map. 
        :type   connector_colour:   str or equivalent colour map object
        :param  connector_points:   [main_x, main_y, window_x, window_y], list of values indicating where 
                                        the connection lines between the main figure and inset window are attached 
                                        on the x and y axes. Minimum and Maximum values are 0 and 1. 
                                        If None the points are automatically chosen. 
        :type   connector_points:   None (by default) or list
        :param  window_zorder:      tuple of zorder values of the overlaid inset window and outlined region on the main figure (inset window, outline)
        :type   window_zorder:      tuple (int or None, int or None)
        :param  norm:               the healpy Gnomonic Projection normalisation for the colour map; 'none', 'log', 'hist' ect....
        :type   norm:               str

        """

        # position of the halo on the sky of the halo
        theta, phi = hp.vec2ang(zoom_vec, lonlat=True) # lonlat=True => [degrees]

        # make zoomed in region 
        zoom_sidelength = [(pix_res*unyt.arcmin * xy_npix[0]).to(unyt.degree), (pix_res*unyt.arcmin * xy_npix[1]).to(unyt.degree)] #degree

        print(f"+ zoomed region ({zoom_sidelength[0]} x {zoom_sidelength[1]})")
        #dset_zoom=hp.gnomview(dset, rot=[theta,phi], xsize=xy_npix[0], ysize=xy_npix[1], reso=pix_res, min=pix_min, max=pix_max, cmap=colour_map, norm='log', return_projected_map=True, no_plot=True) # array for zoomed region on map
        dset_zoom=hp.gnomview(dset, rot=[theta,phi], xsize=xy_npix[0], ysize=xy_npix[1], reso=pix_res, min=pix_min, max=pix_max, norm='log',cmap=cmap, return_projected_map=True, no_plot=True) # array for zoomed region on map

        # make basic gnom object
        gnom_obj = hp.projector.GnomonicProj(rot=[theta,phi], xsize=xy_npix[0], ysize=xy_npix[1], reso=pix_res) # create projection object so we can overlay haloes 
        gnom_x, gnom_y = gnom_obj.ang2xy(theta, phi,lonlat=True) #positions in the gnomietric plane
        gnom_i, gnom_j = gnom_obj.xy2ij(x=gnom_x, y=gnom_y) #pixels in the gnomietric plane
        axes_extent = [gnom_obj.get_extent()[0],
                       gnom_obj.get_extent()[1],
                       gnom_obj.get_extent()[2],
                       gnom_obj.get_extent()[3]]


        Lx=gnom_obj.get_extent()[1] - gnom_obj.get_extent()[0]
        dx=np.radians((scale_bar[0]*unyt.arcmin).to_value(unyt.degree))
        x0 = gnom_obj.get_extent()[0] + (Lx*0.05)
        x1=x0+dx
        y0=gnom_obj.get_extent()[2] + Lx*0.05
        y1=y0

        axins = fig.axes[0].inset_axes(window_vec)
        axins.set_zorder(window_zorder[0])

        #img = axins.imshow(np.flip(dset_zoom, axis=1), origin='lower', cmap=colour_map, norm=norm, vmin=pix_min, vmax=pix_max, extent=axes_extent)
        img = axins.imshow(np.flip(dset_zoom, axis=1), origin='lower',  norm=norm, vmin=pix_min, vmax=pix_max, extent=axes_extent, cmap=cmap)
        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)

        if text_size is None:
           text_size =mpl.rcParams['axes.labelsize']

        if window_text is not None:
            if text_background is True:
                t = axins.text(
                    1-0.975,0.975,
                    window_text,
                    va='top', ha='left', color=text_colour,
                    size=text_size, weight="bold", 
                    transform = axins.transAxes,
                    path_effects=text_path_effects,   
                    bbox=dict(
                        facecolor='lightgrey',
                        edgecolor='none',
                        alpha=0.5,
                        boxstyle='round,pad=0.3'
                        )
                )
            else:
                t = axins.text(
                    1-0.975,0.975,
                    window_text,
                    va='top', ha='left', color=text_colour,
                    size=text_size, weight="bold", 
                    transform = axins.transAxes,
                    path_effects=text_path_effects,   
                )

        axins.plot([x0+scale_xy_shift[0],x1+scale_xy_shift[0]], [y0+scale_xy_shift[1],y1+scale_xy_shift[1]], linewidth=1., color=scale_bar[-1])

        axins.text(
            (x0+0.5*(x1-x0))+scale_xy_shift[0],
            ((y0)+0.001)+scale_xy_shift[1],
            scale_bar[1],
            #r"$60^{\circ}$",
            ha='center',
            va='bottom',
            color=scale_bar[-1],
            size=text_size,
            weight="bold",
            path_effects=[path_effects.withStroke(linewidth=1., foreground="black"), path_effects.Normal()], 
        )
        # zoom region (in main image coordinates)
        x1, x2 = axes_extent[0], axes_extent[1]
        y1, y2 = axes_extent[2], axes_extent[3]

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        # draw rectangle + connector lines automatically
        lon0=theta[0]
        lat0=phi[0]
        
        xlon, xlat = return_circle_mollweide(lon0, lat0, radius_deg=zoom_sidelength[0].to_value(unyt.degree)/2, n=3600)
        ylon, ylat = return_circle_mollweide(lon0, lat0, radius_deg=zoom_sidelength[1].to_value(unyt.degree)/2, n=3600)

        #cosang = (
        #    np.sin(np.deg2rad(lat0))*np.sin(xlat)
        #    + np.cos(np.deg2rad(lat0))*np.cos(xlat)
        #    * np.cos(xlon - np.deg2rad(lon0))
        #)
        #ang = np.rad2deg(np.arccos(np.clip(cosang, -1, 1)))
        #r=zoom_sidelength[0].to_value(unyt.degree)/2
        #fig.axes[0].plot(lon, lat, color="silver", lw=1, linestyle='--')

        axins.spines['bottom'].set_color(connector_colour)
        axins.spines['top'].set_color(connector_colour)
        axins.spines['left'].set_color(connector_colour)
        axins.spines['right'].set_color(connector_colour)

        # rectangle coordinates in main axes coordinates
        rx1, rx2 = np.min(xlon), np.max(xlon)
        ry1, ry2 = np.min(ylat), np.max(ylat)

        rect = Rectangle(
            (rx1, ry1),
            rx2-rx1,
            ry2-ry1,
            edgecolor=connector_colour,
            facecolor="none",
            zorder=window_zorder[-1],
            transform=fig.axes[0].transData
        )
        fig.axes[0].add_patch(rect)

        # inset location in figure coordinates
        bbox = axins.get_position()

        inset_centre = np.array([
            bbox.x0 + bbox.width/2,
            bbox.y0 + bbox.height/2
        ])

        # zoom rectangle centre in figure coordinates
        rect_centre_disp = fig.axes[0].transData.transform(
            [(rx1+rx2)/2, (ry1+ry2)/2]
        )

        rect_centre_fig = fig.transFigure.inverted().transform(rect_centre_disp)

        direction = inset_centre - rect_centre_fig

        if connector_points is None:

            # Choose rectangle side based on direction
            if abs(direction[0]) > abs(direction[1]):
                # inset is mostly left/right
                if direction[0] > 0:
                    rect_points = [
                        (rx2, ry2),
                        (rx2, ry1)
                    ]
                else:
                    rect_points = [
                        (rx1, ry2),
                        (rx1, ry1)
                    ]

                # connect to closest inset vertical edge
                if direction[0] > 0:
                    inset_points = [
                        (0, 1),
                        (0, 0)
                    ]
                else:
                    inset_points = [
                        (1, 1),
                        (1, 0)
                    ]

            else:
                # inset is mostly above/below
                if direction[1] > 0:
                    rect_points = [
                        (rx1, ry2),
                        (rx2, ry2)
                    ]
                else:
                    rect_points = [
                        (rx1, ry1),
                        (rx2, ry1)
                    ]

                # connect to closest inset horizontal edge
                if direction[1] > 0:
                    inset_points = [
                        (0, 0),
                        (1, 0)
                    ]
                else:
                    inset_points = [
                        (0, 1),
                        (1, 1)
                    ]

        else:
            rect_points = (connector_points[0], connector_points[1])
            inset_points = (connector_points[2], connector_points[3])
        # Create connectors
        for (xp, yp), (xi, yi) in zip(rect_points, inset_points):

            con = ConnectionPatch(
                xyA=(xp, yp),
                coordsA=fig.axes[0].transData,
                xyB=(xi, yi),
                coordsB=axins.transAxes,
                color=connector_colour,
                zorder=window_zorder[-1],
            )

            fig.add_artist(con)
        
        if return_img==True:
            return img, axins


