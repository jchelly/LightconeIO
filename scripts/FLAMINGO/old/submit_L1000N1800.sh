#!/bin/bash

if [ "$#" -ne 1 ] ; then
  echo "Usage: ./submit_L1000N1800.sh run_name"
  exit 1
fi

run_name="${1}"
sim="L1000N1800/${run_name}"
if [ ! -d /cosma8/data/dp004/flamingo/Runs/${sim} ] ; then
  echo No directory found for run ${sim}
  exit 1
fi

# Submit job to combine maps
combine_job_id=`sbatch --parsable -J "${run_name}" --array=0-1 ./combine_L1000N1800.sh`
require_combined="--dependency=aftercorr:${combine_job_id}"

# Submit job to correct combined maps
correct_job_id=`sbatch --parsable -J "${run_name}" --array=0-1 ${require_combined} ./correct_L1000N1800.sh`
require_corrected="--dependency=aftercorr:${correct_job_id}"

# Submit job to downsample combined maps
downsample_job_id=`sbatch --parsable -J "${run_name}" --array=0-1 ${require_corrected} ./downsample_L1000N1800.sh`
