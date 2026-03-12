import h5py
import numpy as np
from numba import jit
from unyt import g, cm, mp, erg, s, photons




# The SOAP xray calculator modifed to hanlde multiple redshifts at once

class XrayCalculator_LC:
    def __init__(self, redshifts, table, bands, observing_types):
        if bands == None:
            print('Please specify the band you would like to generate emissivities for\n \
                Using the "band = " keyword\n\n \
                Available options are:\n \
                "erosita-low" (0.2-2.3 keV)\n \
                "erosita-high" (2.3-8.0 keV)\n \
                "ROSAT" (0.5-2.0 keV)')
            raise KeyError
        
        if observing_types == None:
            print('Please specify whether you would like to generate photon or energie emissivities\n \
                Using the "observing_type = " keyword\n\n \
                Available options are:\n \
                "energies_intrinsic"\n \
                "photons_intrinsic"')
            raise KeyError      

        # we do not allow restframe X-ray observation types

        observing_types, bands = self.check_for_restframe(observing_types, bands)


        if (bands != None) & (observing_types != None):
            assert len(bands) == len(observing_types)
        
        self.tables = self.load_all_tables(redshifts, table, bands, observing_types)
        

        self.observation_type_luminosities_cgs_units={
            'photons_intrinsic':photons * s**-1,    
            'energies_intrinsic':erg * s**-1,    
            'photons_convolved':photons * cm**2 * s**-1,    
            'energies_convolved':erg *cm**2 * s**-1
            }
        

    def check_for_restframe(self, observing_types, bands):
        """
        remove the restframe X-ray observation types and corresponding bands
        """
        for i, xray_type in enumerate(observing_types):
            if xray_type in ["energies_intrinsic_restframe", "photons_intrinsic_restframe"]:
                del observing_types[i]
                del bands[i]
        return observing_types, bands

        

    def load_all_tables(self, redshifts, table, bands, observing_types):
        '''
        Load the x-ray tables for the specified bands and observing-types

        Params:
            redshifts: of the particles 
            table: either a path to the table to be read or a dictionary containing the tables themselves
            bands, observing_types: the bands and observation types, within the band to add to tables.
        '''
        if type(table_h5)==str:
            # given a table path, attempt to read the table:
            try:
                table = h5py.File(table_path, "r")
            except ValueError as e:
                raise Exception("You must pass a working x-ray table path") from e
        elif (type(table) != dict) or (type(table_h5)==h5py._hl.files.File):
            # table is not recognisable ....
            raise Exception("You must pass a working x-ray table path, dictionary or h5py.File object")


        self.redshift_bins = table['Bins']['Redshift_bins']
        idx_z, _= self.get_index_1d(self.redshift_bins, redshifts)
        ############ make it always load min redshift index value. 
        #min_idx_z = np.min(idx_z)
        min_idx_z = 0
        ############
        max_idx_z = np.max(idx_z) + 2

        self.He_bins = table['Bins']['He_bins']
        self.missing_elements = table['Bins']['Missing_element']
        self.element_masses = table['Bins']['Element_masses']

        self.density_bins = table['Bins']['Density_bins']
        self.temperature_bins = table['Bins']['Temperature_bins']
        self.redshift_bins = table['Bins']['Redshift_bins']

        self.log10_solar_metallicity = table['Bins']['Solar_metallicities']
        self.solar_metallicity = np.power(10, self.log10_solar_metallicity)


        tables = {}
        for band in bands:
            tables[band] = {}
            for observing_type in observing_types:
                temp = table[band][observing_type][int(min_idx_z):int(max_idx_z), :, :, :, :].astype(np.float32)
                # temp = np.swapaxes(temp, 2, 4)
                # self.table_shape = temp.shape[:-1]
                # temp = temp.reshape((temp.shape[0] * temp.shape[1] * temp.shape[2] * temp.shape[3], temp.shape[4]))
                tables[band][observing_type] = temp

        return tables

        
    @staticmethod
    @jit(nopython = True)
    def get_index_1d(bins, subdata):
        '''
        Find the closest bin index below the specified value, and the relative offset compared to that bin.
        '''
        eps = 1e-4
        delta = (len(bins) - 1) / (bins[-1] - bins[0])

        idx = np.zeros_like(subdata)
        dx = np.zeros_like(subdata, dtype = np.float32)
        for i, x in enumerate(subdata):
            if x < bins[0] + eps:
                # We are below the first element
                idx[i] = 0
                dx[i] = 0
            elif x < bins[-1] - eps:
                # Normal case
                idx[i] = int((x - bins[0]) * delta)
                dx[i] = (x - bins[int(idx[i])]) * delta
            else:
                # We are after the last element
                idx[i] = len(bins) - 2
                dx[i] = 1
            
        return idx, dx

    @staticmethod
    @jit(nopython = True)
    def get_index_1d_irregular(bins, subdata):
        '''
        Find the closest bin index below the specified value, and the relative offset compared to that bin.
        Unlike get_index_1d, this allows for irregular bin spacing
        '''
        eps = 1e-6
        idx = np.zeros_like(subdata)
        dx = np.zeros_like(subdata, dtype = np.float32)

        for i, x in enumerate(subdata):
            if x < bins[0] + eps:
                idx[i] = 0
                dx[i] = 0
            elif x < bins[-1] - eps:
                min_idx = -1

                '''
                Do this the hard way: Search the table
                for the smallest index i in bins[i] such
                that table[i] < x
                '''
                for j in range(len(bins)):
                    if x - bins[j] <= 0:
                        # Found the first entry that is larger than x, go back by 1
                        min_idx = j - 1
                        break

                idx[i] = min_idx
                dx[i] = (x - bins[min_idx]) / (bins[min_idx + 1] - bins[min_idx])
            else:
                idx[i] = len(bins) - 2
                dx[i] = 1

        return idx, dx

    @staticmethod
    # @jit(nopython = True)
    def get_table_interp(idx_z, idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, X_Ray, abundance_to_solar):
        '''
        4D interpolate the x-ray table for each traced metal
        Scale the metals with their respective relative solar abundances
        Add the metals to the background case without metals
        '''

        f_n_T = np.zeros((t_nH.shape[0], X_Ray.shape[1]), dtype = np.float32)

        f_n_T += (t_nH * t_He * t_T * t_z)[:, None] * X_Ray[idx_z, idx_he, :, idx_T, idx_n]
        f_n_T += (t_nH * t_He * d_T * t_z)[:, None] * X_Ray[idx_z, idx_he, :, idx_T + 1, idx_n]
        f_n_T += (t_nH * d_He * t_T * t_z)[:, None] * X_Ray[idx_z, idx_he + 1, :, idx_T, idx_n]
        f_n_T += (d_nH * t_He * t_T * t_z)[:, None] * X_Ray[idx_z, idx_he, :, idx_T, idx_n + 1]


        f_n_T += (t_nH * d_He * d_T * t_z)[:, None] * X_Ray[idx_z, idx_he + 1, :, idx_T + 1, idx_n]
        f_n_T += (d_nH * t_He * d_T * t_z)[:, None] * X_Ray[idx_z, idx_he, :, idx_T + 1, idx_n + 1]
        f_n_T += (d_nH * d_He * t_T * t_z)[:, None] * X_Ray[idx_z, idx_he + 1, :, idx_T, idx_n + 1]
        f_n_T += (d_nH * d_He * d_T * t_z)[:, None] * X_Ray[idx_z, idx_he + 1, :, idx_T + 1, idx_n + 1]


        f_n_T += (t_nH * t_He * t_T * d_z)[:, None] * X_Ray[idx_z + 1, idx_he, :, idx_T, idx_n]
        f_n_T += (t_nH * t_He * d_T * d_z)[:, None] * X_Ray[idx_z + 1, idx_he, :, idx_T + 1, idx_n]
        f_n_T += (t_nH * d_He * t_T * d_z)[:, None] * X_Ray[idx_z + 1, idx_he + 1, :, idx_T, idx_n]
        f_n_T += (d_nH * t_He * t_T * d_z)[:, None] * X_Ray[idx_z + 1, idx_he, :, idx_T, idx_n + 1]

        f_n_T += (t_nH * d_He * d_T * d_z)[:, None] * X_Ray[idx_z + 1, idx_he + 1, :, idx_T + 1, idx_n]
        f_n_T += (d_nH * t_He * d_T * d_z)[:, None] * X_Ray[idx_z + 1, idx_he, :, idx_T + 1, idx_n + 1]
        f_n_T += (d_nH * d_He * t_T * d_z)[:, None] * X_Ray[idx_z + 1, idx_he + 1, :, idx_T, idx_n + 1]
        f_n_T += (d_nH * d_He * d_T * d_z)[:, None] * X_Ray[idx_z + 1, idx_he + 1, :, idx_T + 1, idx_n + 1]



        # Add each metal contribution individually
        f_n_T_Z_temp = np.power(10, f_n_T[:, -1], dtype=np.float64)
        for j in range(f_n_T.shape[1] - 1):
            f_n_T_Z_temp += np.power(10, f_n_T[:, j]) * abundance_to_solar[:, j]

        f_n_T_Z = np.log10(f_n_T_Z_temp)
        



        return f_n_T_Z
    
    def find_indices(self, densities, temperatures, element_mass_fractions, masses, redshifts, fill_value = 0):
        '''
        Check interpolation table bounds
        Compute all interpolation bin indices, and the offsets from those bins
        Compute all the indices for the flattened x-ray table
        '''
        scale_factors = 1 / (1 + redshifts)
        data_n = np.log10(element_mass_fractions[:, 0] * (1 / scale_factors**3) * densities.to(g * cm**-3) / mp)
        data_T = np.log10(temperatures)
        volumes = (masses.astype(np.float64) / ((1 / scale_factors**3) * densities.astype(np.float64))).to(cm**3)

        # Create density mask, round to avoid numerical errors
        density_mask = (data_n >= np.round(self.density_bins.min(), 1)) & (data_n <= np.round(self.density_bins.max(), 1))
        # Create temperature mask, round to avoid numerical errors
        temperature_mask = (data_T >= np.round(self.temperature_bins.min(), 1)) & (data_T <= np.round(self.temperature_bins.max(), 1))

        # Combine masks
        joint_mask = density_mask & temperature_mask


        # Check if within density and temperature bounds
        density_bounds = np.sum(density_mask) == density_mask.shape[0]
        temperature_bounds = np.sum(temperature_mask) == temperature_mask.shape[0]
        if ~(density_bounds & temperature_bounds):
            #If no fill_value is set, return an error with some explanation
            if fill_value == None:
                raise ValueError(
                        "Temperature or density are outside of the interpolation range and no fill_value is supplied\n \
                        Temperature ranges between log(T) = 5 and log(T) = 9.5\n \
                        Density ranges between log(nH) = -8 and log(nH) = 6\n \
                        Set the kwarg 'fill_value = some value' to set all particles outside of the interpolation range to 'some value'\n \
                        Or limit your particle data set to be within the interpolation range"
                    )
            else:
                pass
        
        #get individual mass fraction
        mass_fraction = element_mass_fractions[joint_mask]

        # Find redshift offsets
        idx_z, dx_z = self.get_index_1d(self.redshift_bins, redshifts[joint_mask])
        idx_z = idx_z.astype(int)

        # Find density offsets
        idx_n, dx_n = self.get_index_1d(self.density_bins, data_n[joint_mask])
        idx_n = idx_n.astype(int)

        # Find temperature offsets
        idx_T, dx_T = self.get_index_1d(self.temperature_bins, data_T[joint_mask])
        idx_T = idx_T.astype(int)

        # Calculate the abundance wrt to solar
        abundances = (mass_fraction / np.expand_dims(mass_fraction[:, 0], axis = 1)) * (self.element_masses[0] /  np.array(self.element_masses))

        # Calculate abundance offsets using solar abundances
        abundance_to_solar = abundances / self.solar_metallicity

        # Add columns for Calcium and Sulphur and move Iron to the end
        abundance_to_solar = np.c_[abundance_to_solar[:, :-1], abundance_to_solar[:, -2], abundance_to_solar[:, -2], abundance_to_solar[:, -1]] 

        #Find helium offsets
        idx_he, dx_he = self.get_index_1d_irregular(self.He_bins, np.log10(abundances[:, 1]))
        idx_he = idx_he.astype(int)


        t_z = 1 - dx_z
        d_z = dx_z

        # Compute temperature offset relative to bin
        t_T = 1 - dx_T
        d_T = dx_T

        # Compute density offset relative to bin
        t_nH = 1 - dx_n
        d_nH = dx_n

        # Compute Helium offset relative to bin
        t_He = 1 - dx_he
        d_He = dx_he

        return idx_z, idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n


    def interpolate_X_Ray(self, idx_z, idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n, bands = None, observing_types = None, fill_value = None):
        '''
        Main function
        Calculate the particle emissivities through interpolation
        Convert to luminosity using the particle volume
        '''
        # Initialise the emissivity array which will be returned
        emissivities = np.zeros((joint_mask.shape[0], len(bands)), dtype = float)
        luminosities = np.zeros_like(emissivities)
        emissivities[~joint_mask] = fill_value

        # Interpolate the table for each specified band
        for i_interp, band, observing_type in zip(range(len(bands)), bands, observing_types):

            emissivities[joint_mask, i_interp] = self.get_table_interp(idx_z, idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, self.tables[band][observing_type], abundance_to_solar[:, 2:])

            # Convert from erg cm^3 s^-1 to erg cm^-3 s^-1
            # To do so we multiply by nH^2, this is the actual nH not the nearest bin
            # It allows to extrapolate in density space without too much worry
            # log(emissivity * nH^2) = log(emissivity) + 2*log(nH)
            emissivities[joint_mask, i_interp] += 2*data_n[joint_mask]

            luminosities[joint_mask, i_interp] = np.power(10, emissivities[joint_mask, i_interp]) * volumes[joint_mask]

        # get X-ray observation type units
        luminosities_cgs_unyts = self.observation_type_luminosities_cgs_units[observing_types[0]]

        if 'energies' in observing_types[0]:
            return luminosities * luminosities_cgs_unyts
        elif 'photon' in observing_types[0]:
            return luminosities * luminosities_cgs_unyts

def xray_map_names(observation_band, observation_type):
    dataset_names={
        "ROSAT_photons_intrinsic":"XrayROSATIntrinsicPhotons",
        "ROSAT_photons_convolved":"XrayROSATConvolvedPhotons",
        "ROSAT_energies_intrinsic":"XrayROSATIntrinsicEnergies",
        "ROSAT_energies_convolved":"XrayROSATConvolvedEnergies",
        "erosita-high_photons_intrinsic":"XrayErositaHighIntrinsicPhotons",
        "erosita-high_photons_convolved":"XrayErositaHighConvolvedPhotons",
        "erosita-high_energies_intrinsic":"XrayErositaHighIntrinsicEnergies",
        "erosita-high_energies_convolved":"XrayErositaHighConvolvedEnergies",
        "erosita-low_photons_intrinsic":"XrayErositaLowIntrinsicPhotons",
        "erosita-low_photons_convolved":"XrayErositaLowConvolvedPhotons",
        "erosita-low_energies_intrinsic":"XrayErositaLowIntrinsicEnergies",
        "erosita-low_energies_convolved":"XrayErositaLowConvolvedEnergies"
    }
    return dataset_names[observation_band+'_'+observation_type]



COMBINED_XRAY_EMISSIVITY_TABLEe_FILENAME = "/cosma8/data/dp004/flamingo/Tables/Xray/X_Ray_table_combined.hdf5"
