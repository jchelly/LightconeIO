#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/L2800N5040/untar_%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 12:00:00
#

name=${SLURM_JOB_NAME}
lightcone_nr=${SLURM_ARRAY_TASK_ID}

module purge
module load gnu_comp/14.1.0 openmpi

srcdir="/cosma8/data/dp004/flamingo/Runs/L2800N5040/${name}/lightcones/lightcone${lightcone_nr}_particles/"
destdir="/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/L2800N5040/${name}/"

\mkdir -p ${destdir}

cd ${destdir} && mpirun -np 20 /cosma/home/dp004/jch/Codes/parallel_tasks/build/parallel_tasks 0 80 \
    "tar xvfz ${srcdir}/lightcone0_%04d.tar.gz"
