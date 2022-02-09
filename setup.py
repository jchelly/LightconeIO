#!/usr/bin/env python

from setuptools import setup

setup(name='SWIFT Lightcone I/O',
      version='0.1',
      description='Code for reading SWIFT lightcone output',
      author='John Helly',
      author_email='j.c.helly@durham.ac.uk',
      packages=['lightcone_io',],
      scripts=['bin/lightcone_io_index_particles.py',
               'bin/lightcone_io_combine_maps.py',
               'bin/lightcone_io_compare_maps.py',],
     )
