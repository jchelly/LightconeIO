__all__ = ["ShellArray", "Shell", "HealpixMap", "LightconeParticles", "LightconeParticleType"]

# Use setuptools_scm to get version from git tags
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("lightcone_io")
except PackageNotFoundError:
    __version__ = "unknown"

# Classes for reading lightcone particle data
from .particle_reader import IndexedLightconeParticleType as LightconeParticleType
from .particle_reader import IndexedLightcone as LightconeParticles

# Classes for reading healpix maps
from .healpix_maps import Shell, ShellArray, HealpixMap
