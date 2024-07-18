#!/bin/env python

import sys

import lightcone_io.healpix_maps as hm

def main():

    usage="""
    python3 ./lightcone_io_compare_maps.py indir1 indir2 lightcone_nr shell_nr

    indir1      : directory containing the first set of maps
    indir2      : directory containing the second set of maps
    lightcone_nr: which lightcone to compare
    shell_nr    : which shell to compare
"""

    if len(sys.argv) != 5:
        print(usage)
        sys.exit(0)

    indir1       = sys.argv[1]
    indir2       = sys.argv[2]
    lightcone_nr = int(sys.argv[3])
    shell_nr     = int(sys.argv[4])    
    basename = "lightcone%d" % lightcone_nr

    # Find full list of datasets
    fname1 = "%s/%s_shells/shell_%d/%s.shell_%d.0.hdf5" % (indir1, basename, shell_nr, basename, shell_nr)
    map_names1 = hm.get_map_names(fname1)
    fname2 = "%s/%s_shells/shell_%d/%s.shell_%d.0.hdf5" % (indir2, basename, shell_nr, basename, shell_nr)
    map_names2 = hm.get_map_names(fname2)
    datasets = set(map_names1 + map_names2)

    for dataset in datasets:
        maps_match = hm.compare_healpix_maps(indir1, indir2, basename, shell_nr, dataset)
        if maps_match != True:
            raise Exception("Map %s for %s shell %d does not match!" % (dataset, basename, shell_nr))
        else:
            print("Map %s for %s shell %d is ok" % (dataset, basename, shell_nr))
