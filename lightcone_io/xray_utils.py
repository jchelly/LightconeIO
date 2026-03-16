import os
import h5py
import numpy as np
import unyt
import re
from astropy.cosmology import w0waCDM


# create cosmology object with snapshot information that is needed to filter the X-ray particles
class Snapshot_Cosmology_For_Lightcone: 
    def __init__(self, snapshot_root_dir):

        """
        Make a cosmology object from the simuations snapshot cosmology data with useful functions
        for the particle lightcones.
        """
        
        # try access snapshot file
        self.all_snapshot_filenames=self.get_snapshot_filename(snapshot_root_dir)

        # to be consistent use the lowest redshift (highest number) snapshot. 
        snapshot_path = self.all_snapshot_filenames[-1] 
        
        # collect cosmology info from hdf5
        self.raw_cosmo = self.cosmology_from_hdf5(snapshot_path)
        
        # collect internal units 
        self.internal_units = self.get_internal_units(snapshot_path)

        # make cosmology object
        self.COSMO = w0waCDM(
            H0=unyt.unyt_quantity(self.raw_cosmo['H0 [internal units]'][0], units='km / Mpc / s').to("1/s").to_astropy(),
            Om0=self.raw_cosmo['Omega_m'][0],
            Ode0=self.raw_cosmo['Omega_lambda'][0],
            Ob0=self.raw_cosmo['Omega_b'][0],
            w0=self.raw_cosmo['w_0'][0],
            wa=self.raw_cosmo['w_a'][0],
            Tcmb0=unyt.unyt_quantity(self.raw_cosmo['T_CMB_0 [K]'][0], units=unyt.K),
        )
        
        # needed for filtering of recently heated gas particles
        self.AGN_delta_T_K = self.get_AGN_delta_T_K(snapshot_path)
    
    def get_snapshot_filename(self, rootdir):
        """
        Return path to lowest redshift snapshot file.
        Relies on FLAMINGO snapshot virtual files to be named as: 
            /snapshots/flamingo_numb/flamingo_numb.hdf5

        Params:
            rootdir: path to directory with flamingo snapshots

        """
        try:
            regex = re.compile('(flamingo_00[0-9][0-9][.]hdf5$)')
            filenames=[]
            for root, dirs, files in os.walk(rootdir):
                for file in files:
                    if regex.match(file):
                        filenames.append( os.path.join(root, file))
            filenames.sort()
            return filenames
        except:
            raise ValueError("cannot locate snapshot from root directory. ")


    def cosmology_from_hdf5(self, snapshot_path):
        with h5py.File(snapshot_path, "r") as sn:
            raw_cosmo = {}
            for k,v in sn['Cosmology'].attrs.items():
                raw_cosmo[f'{k}']=v
        return raw_cosmo
    
    def get_internal_units(self, snapshot_path):
        internal_units={}
        with h5py.File(snapshot_path, "r") as sn:
            for k, v in sn['InternalCodeUnits'].attrs.items():
                internal_units[f'{k}'] = v
        return internal_units


    def z2Myr(self, z):
        """
        Age of universe (time from big bang) to input redshift. 
        Returns a unyt_array [Myrs]
        """
        t_age=self.COSMO.age(z).to("Myr")
        return unyt.unyt_array.from_astropy(t_age)
    
    def redshift_with_time_offset(self, z, dt):
        """
        params:
            z: redshift of observer
            dt: length of time [Myr]

        returns 
            The redshift equal to the input redshift - length of time (dt)
        """
        time_from_bb = self.COSMO.age(z).to('Myr')
        time_with_dt_bb = (time_from_bb.to_value("Myr") - dt)*unyt.Myr
        z_min = z_at_value(self.COSMO.age, time_with_dt_bb.to_astropy())
    
        return z_min.value

    def z2r(self, z):
        """
        Returns the comoving radius for a given redshift.
        """
        return unyt.unyt_array.from_astropy(self.COSMO.comoving_distance(z)).to("Mpc")
    
    def shell_comoving_raddii(self, redshift_filename):
        """
        Params:
            redshift_filename: path to .txt file with redshift shell
        Returns 
            Array of comoving radii (inner, outer)
        """
        try:
            self.shell_redshifts = np.loadtxt(redshift_filename, delimiter=",")
            zmin = self.shell_redshifts[:,0]
            zmax = self.shell_redshifts[:,1]
            comoving_radii = unyt.unyt_array(np.zeros((len(zmin), 2)), units="Mpc")
            comoving_radii[:, 0]+=self.z2r(zmin)
            comoving_radii[:, 1]+=self.z2r(zmax)
        except:
            raise ValueError("cannot access shell redshifts .txt file")
        
        return comoving_radii
        

    def get_AGN_delta_T_K(self, snapshot_path):
        """
        return the Delta_T_AGN param required to identify recently heated particles
        """
        # 'AGN_delta_T_K' Change in temperature to apply to the gas particle in an AGN feedback event [K]
        with h5py.File(snapshot_path, "r") as sn:
            if 'EAGLEAGN:AGN_delta_T_K' in sn['Parameters'].attrs: # check if thermal mode AGN_delta_T_K exists
                AGN_delta_T_K = np.float64(sn['Parameters'].attrs['EAGLEAGN:AGN_delta_T_K'])*unyt.K
            elif 'SPINJETAGN:AGN_delta_T_K' in sn['Parameters'].attrs: # check if Jet mode AGN_delta_T_K exists
                AGN_delta_T_K = np.float64(sn['Parameters'].attrs['SPINJETAGN:AGN_delta_T_K'])*unyt.K
            else: # cannot find thermal or Jet mode AGN_delta_T_k param 
                raise NameError("\n!!!\nCannot find AGN_delta_T_K value:\n\t'EAGLEAGN:AGN_delta_T_K' or 'SPINJETAGN:AGN_delta_T_K' not in snapshot 'Parameters'\n!!!")
                AGN_delta_T_K=-1*unyt.K # set as an error value

        return AGN_delta_T_K
    
    def lightcone_volume(self, comoving_inner_radius, comoving_outer_radius, use_redshift=False, phi=None):
        """
        Compute the comoving volume of the lightcone given some inner and outer radius

        params: 
            comoving_inner_radius, comoving_outer_radius: the radius of the lightcone in Mpc
            use_redshift [boolean]: if False, then radii are in comoving distances. if True, then
                                    input radii are redshift values. 
            phi: 1/2 angle of the cones apature [radian]
        """
        if use_redshift: 
            # redshift -> comoving distance
            comoving_inner_radius = self.COSMO.comoving_distance(comoving_inner_radius)
            comoving_outer_radius = self.COSMO.comoving_distance(comoving_outer_radius)
        
        if phi is None:
            shell_volume = (4/3)*np.pi * (comoving_outer_radius**3 - comoving_inner_radius**3)
        else:
            r_outer=comoving_outer_radius*(1-np.cos(phi))
            r_inner=comoving_inner_radius*(1-np.cos(phi))
            shell_volume = (2/3)*np.pi * ((comoving_outer_radius**2 * r_outer) - (comoving_outer_radius**2 * r_outer))
        
        return shell_volume


class Xray_Filter:
    def __init__(self, particle_data, numb_particles, remove_SF=True, log_T_min_K=5, remove_rhp=True, cosmo=None):

        
        # create a mask that tracks all particles which can be used for X-ray values
        self.KEEP_FOR_XRAY=np.ones(numb_particles, dtype=bool)

        if remove_SF:
            if "StarFormationRates" in particle_data:
                self.filter_by_StarFormationRates(particle_data["StarFormationRates"].value)
            else:
                raise ValueError("particle StarFormationRates not found ....")
        
        if "Temperatures" in particle_data:
            self.filter_by_Temperatures(particle_data["Temperatures"].to_value("K"), log_T_min_K)
        else:
            raise ValueError("particle Temperatures not found ....")

        if remove_rhp and cosmo is not None:
            # check all the needed properties exist in particle data
            for prop_name in ["ExpansionFactors", "LastAGNFeedbackScaleFactors", "Temperatures", "Densities"]:
                if prop_name not in particle_data:
                    ValueError("particle {required_property} not found ....".format(required_property=prop_name))

            if cosmo is not None:


                # identify all recently heated particles and keep mask for future use
                is_recently_heated = self.identify_recently_heated(
                    cosmo,
                    scalefactors=particle_data["ExpansionFactors"][self.KEEP_FOR_XRAY].value,
                    last_agn_feedback_scalefactors=particle_data["LastAGNFeedbackScaleFactors"][self.KEEP_FOR_XRAY].value,
                    temperatures=particle_data["Temperatures"][self.KEEP_FOR_XRAY], # keep units 
                    densities=particle_data["Densities"][self.KEEP_FOR_XRAY].to("g * cm**-3"), # convert the units
                    RHP_filter_max_time_Myr=70,  
                    RHP_filter_log_density_cm3=-2.25)

                self.numb_recently_heated = np.sum(is_recently_heated)

                # update mask 
                self.KEEP_FOR_XRAY[self.KEEP_FOR_XRAY][is_recently_heated]=0
        
        else:
            print("Need to provide cosmology information to filter recently heated particles")
            



    def filter_by_StarFormationRates(self, sfr):
        """
        remove all star forming gas particles & update mask
        
        Params
            sfr: gas particle star formation rates

        """
        self.KEEP_FOR_XRAY[sfr>0.]=0

    def filter_by_Temperatures(self, temperatures, log_T_min_K):
        """
        keep only `hot' gas particles & update mask
        
        Params
            temperatures: temperatures of gas particles
            log_T_min_K: minimum temperature to be considered a hot gas particle
        
        """
        self.KEEP_FOR_XRAY[temperatures<10**log_T_min_K]=0


    def time_from_last_AGN_feedback(self, 
        scalefactors, last_agn_feedback_scalefactors, 
        f_z2t, handle_negatives=0.):
        """
        Compute the difference in time from current scalefactor and the scalefactor 
        when last heated by AGN feedback. 
        Due to compression on LastAGNFeedbackScaleFactors values, sometimes 
        LastAGNFeedbackScaleFactors > scalefactor, therefore we must allow negative values. 

        Params
            scalefactors: scalefactor of gas particles
            last_agn_feedback_scalefactor: scalefactor that particle was last 
                                            directly heated by AGN feedback and -1 
                                            if it has never been heated by AGN 
                                            feedback
            f_z2t: function to give age of universe for a given redshift
            handle_negatives: time difference given to where 
                                last_agn_feedback_scalefactor > scalefactors
        """
    
        # make output array of time differences  
        dt = np.zeros(len(scalefactors), dtype=float) -1.
        dt[last_agn_feedback_scalefactors<0]=999 # cannot ever be recently heated
        dt[last_agn_feedback_scalefactors >= scalefactors] = handle_negatives # assume its due to compression

        mask = (last_agn_feedback_scalefactors < scalefactors) & (last_agn_feedback_scalefactors>0)

        if np.sum(mask)>0:

            z_part = 1./scalefactors[mask] -1.
            z_last_agn_feedback = 1./last_agn_feedback_scalefactors[mask] -1.

            particle_time_from_big_bang_Myr = f_z2t(z_part)
            last_heating_time_from_big_bang_Myr = f_z2t(z_last_agn_feedback)
            
            dt[mask]=particle_time_from_big_bang_Myr.to_value("Myr")-last_heating_time_from_big_bang_Myr.to_value("Myr")
            
        return dt


    def identify_recently_heated(
        self, 
        cosmo,
        scalefactors,
        last_agn_feedback_scalefactors, 
        temperatures, 
        densities,
        RHP_filter_max_time_Myr,
        RHP_filter_log_density_cm3=-2.25):
        """
        Identify recently heated particles.
        Due to the BFloat16  lossy compression filter applied to 
        LastAGNFeedbackScaleFactors we cannot reliably determine if a particle 
        was last heated by AGN feedback within the last 15 Myr. 
        Note the minimum density criterion is included to compensate for the uncertanty 
        about LastAGNFeedbackScaleFactors values due to the lossy compression filter used for the 
        particle lightcones.
        To reproduce snapshot selection of recently heated particles:  
            RHP_filter_max_time_Myr=15
            RHP_filter_log_density=None

        
        Params
            cosmo: cosmology object, i.e. Snapshot_Cosmology_For_Lightcone()
            scalefactors: scalefactor of gas particles
            last_agn_feedback_scalefactor: scalefactor that particle was last 
                                            directly heated by AGN feedback and -1 
                                            if it has never been heated by AGN 
                                            feedback
            temperatures (unyt_array): temperatures of gas particles [K]
            densities (unyt_array): densities of gas particles [g/cm^3]
            RHP_filter_max_time_Myr: maximum time in Myrs that has elapsed from the 
                                    LastAGNFeedbackScaleFactors to be considered a recently heated 
                                    particle. If == None then adopt 70 Myr as compromise for uncertanty.
            RHP_filter_log_density: log10(minimum gas density [1/cm^3]), the minimum density 
                                    to be considered a recently heated particle. If == None, then 
                                    do not use density to select recently heated particles.
                                    -2.25 selected as default for McDonald+26 based on 
                                    HYDRO_FIDUCIAL L1_m9 snapshots & ROSAT photon intrinsic maps
        Returns 
            Boolean mask, True if RHP, otherwise False
        """

        
        # find particles that have never been heated by AGN feedback, LastAGNFeedbackScaleFactors==-1
        never_heated = last_agn_feedback_scalefactors<0
        
        # if no particles have been heated by AGN then no need to continue filtering
        npart = len(scalefactors)
        if np.sum(never_heated) == npart:
            return np.zeros(npart, dtype=bool)

        # get temperature range for AGN heating
        AGN_delta_T_K = cosmo.AGN_delta_T_K
        xray_maps_recent_AGN_injection_delta_logT_min=-1 #from IC file
        xray_maps_recent_AGN_injection_delta_logT_max=0.3 #from IC file
        lower_T_K = AGN_delta_T_K*10**xray_maps_recent_AGN_injection_delta_logT_min
        upper_T_K = AGN_delta_T_K*10**xray_maps_recent_AGN_injection_delta_logT_max

        # check if temperatures are within the range expected from recent AGN heating
        is_in_temp_bounds = (temperatures>lower_T_K) & (temperatures<upper_T_K)

        # max time since last heating by AGN feedback and min density criteria
        if RHP_filter_max_time_Myr is None:
            dt_max=70
        else:
            dt_max = RHP_filter_max_time_Myr
        
        if RHP_filter_log_density_cm3 is None:
            # do not select RHP based on density.
            is_above_min_density = np.ones(npart, dtype=bool)
        else:
            min_gas_density_cm3 = 10**RHP_filter_log_density_cm3
            # check minimum density requirent
            is_above_min_density = (densities/unyt.mp).to_value('cm**-3') >= min_gas_density_cm3
        
        # combine masks and clean up
        combined_mask = is_in_temp_bounds & is_above_min_density & np.invert(never_heated)
        
        # if there are no more particles remaining then no need to check time since heating
        if np.sum(combined_mask) == 0:
            return np.zeros(npart, dtype=bool)

        del never_heated
        del is_in_temp_bounds
        del is_above_min_density
        del densities
        del temperatures

        # for remaining particles check the time since they where last heated by AGN feedback
        time_from_last_heating=np.zeros(npart, dtype=float)
        time_from_last_heating[combined_mask==False] = 999. # cannot possibly be recently heated.
        time_from_last_heating[combined_mask] += self.time_from_last_AGN_feedback(
            scalefactors[combined_mask], 
            last_agn_feedback_scalefactors[combined_mask], 
            f_z2t=cosmo.z2Myr,
            handle_negatives=0.)

        # combine masks again to flag recently heated particles
        is_recently_heated = combined_mask & (time_from_last_heating<=RHP_filter_max_time_Myr)

        return is_recently_heated




    

