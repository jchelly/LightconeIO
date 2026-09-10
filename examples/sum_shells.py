#!/bin/env python3
import sys
import os
import h5py
import unyt
import numpy as np
import healpy as hp
import argparse
from lightcone_io.utils import sum_maps


if __name__=="__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("infile_format", type=str,  help="formated path to input files /path/to/filename_{filenumber}.hdf5")
    argparser.add_argument("outfile", type=str,  help="output file that will be written /path/to/filename.hdf5")
    argparser.add_argument("--init_file_numb", type=int,default=-1, help="index of the first file to combined")
    argparser.add_argument("--final_file_numb", type=int,default=-1, help="index of the final file to combined")
    argparser.add_argument("--map_names", type=lambda s: s.split(","), default=["common"], help="comma-seperated list of map dataset names, =common then use all common maps across the shells")
    argparser.add_argument("--file_numbs",type=lambda s: [int(x) for x in s.split(",")], default=[-1], help="comma-seperated list of map file numbers to sum over") 

    args = argparser.parse_args()

    if args.init_file_numb==-1 or args.final_file_numb==-1:
        # no range given, then use the specific numbers given as an input
        if args.file_numbs[0]==-1:
            raise ValueError("file numbers are incorrect")
        else:
            file_numbers = np.asarray(args.file_numbs).astype(int)
        print("\nSumming file:\t ", file_numbers, flush=True)
    else:
        # range given, use all files in this range of numbers
        file_numbers=np.arange(args.init_file_numb, args.final_file_numb+1, 1)
        print(f"\nSumming files:\t {args.init_file_numb} - {args.final_file_numb}", flush=True)
    
    __ = sum_maps(file_numbers, args.infile_format, args.outfile, args.map_names)

