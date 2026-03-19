#!/bin/bash -l
#SBATCH --array=5 # shell number
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -J HYDRO_FIDUCIAL
#SBATCH -o ./logs/smoothed_xray_map.%x.%a.out              
#SBATCH -e ./logs/smoothed_xray_map.%x.%a.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 03:00:00



# NOTE this example is parallelised with MPI. It requires
# h5py to be configured for use with MPI in parallel
# and the use of the VirgoDC repository
# To build the required virtual environment: 
# scripts/virtual_env/make_cosma_env_python3.12.4.sh


module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

simulation_LN="L1000N1800"
simulation_name="${SLURM_JOB_NAME}"

# replace with own lightcone_io virtual env
source /cosma/apps/dp004/${USER}/lightcone_env/bin/activate

# lightcone (or observer) number 
lightcone_nr=0 

# HEALPix map resolution
map_nside=128

# X-ray observation type, determines the units of the output X-ray flux in the map.
observation_type="photons_intrinsic"

output_dir="./test_outputs/${simulation_LN}/${simulation_name}/nside${map_nside}/lightcone${lightcone_nr}_shells"
simulation_dir="/cosma8/data/dp004/flamingo/Runs/${simulation_LN}/${simulation_name}"

mkdir -p "${output_dir}" 

mpirun -- python3 -m mpi4py ./examples/make_xray_map.py ${SLURM_ARRAY_TASK_ID} \
    --nside=${map_nside} \
    --lightcone_nr=${lightcone_nr} \
    --simulation_dir=${simulation_dir} \
    --output_dir=${output_dir} \
    --xray_type=${observation_type} \


