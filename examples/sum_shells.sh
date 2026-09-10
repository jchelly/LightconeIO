#!/bin/bash
#
#SBATCH --nodes=1   #2
#SBATCH --tasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH -J HYDRO_FIDUCIAL
#SBATCH -o ./logs/SumShells.%a.out
#SBATCH -e ./logs/SumShells.%a.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 06:00:00


module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

# format of input files:
infile_format=".FLAMINGO/L1000N1800/${SLURM_JOB_NAME}/nside_4096/lightcone1_shells/lightcone1.shell_{file_nr}.hdf5"

# write new file:
outfile=".FLAMINGO/L1000N1800/${SLURM_JOB_NAME}/nside_4096/lightcone1_shells/lightcone1.Summed_All0to5.hdf5" 

# Specifically want to sum over shell 0 -> 5 shells of a lightcone. Compute the sum total of a map if it is common to each shell
python3 ./lightcone_io/sum_shells.py ${infile_format} ${outfile} \
  --init_file_numb=0 \
  --final_file_numb=5 \
  --map_names=common \



# write new file:
outfile=".FLAMINGO/L1000N1800/${SLURM_JOB_NAME}/nside_4096/lightcone1_shells/lightcone1.Summed_Xray2t9.hdf5" 

# Specifically want to sum every second shell for shell 0 -> 8 shells of a lightcone, but only for select maps 
python3 ./lightcone_io/sum_shells.py ${infile_format} ${outfile} \
  --file_numbs=0,2,4,6,8 \ # use only shells in a comma-seperated list
  --map_names=ComptonY,SmoothedGasMass,BlackHoleMass \ # use only these maps in a comma-seperated list

