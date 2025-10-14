__all__ = ["ShellArray", "Shell", "HealpixMap", "ParticleLightcone", "IndexedLightconeParticleType"]

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
