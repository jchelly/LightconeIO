__all__ = ["ShellArray", "Shell", "HealpixMap", "ParticleLightcone", "IndexedLightconeParticleType", "HaloLightconeFile", "XrayCalculator_LC", "Snapshot_Cosmology_For_Lightcone", "Xray_Filter"]

# Use setuptools_scm to get version from git tags
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("lightcone_io")
except PackageNotFoundError:
    __version__ = "unknown"

# Classes for reading lightcone particle data
from .particle_reader import IndexedLightconeParticleType
from .particle_reader import ParticleLightcone

# Classes for reading healpix maps
from .healpix_maps import Shell, ShellArray, HealpixMap

# Classes for reading halo lightcones
from .halo_reader import HaloLightconeFile

# Classes for computing X-ray values
from .lc_xray_calculator import XrayCalculator_LC

# Classes for using cosmology objects and filtering particles for X-rays
from .xray_utils import Snapshot_Cosmology_For_Lightcone, Xray_Filter
