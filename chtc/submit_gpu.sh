results_dir=results/${1}
log_dir=logs/${1}
mkdir -p ${results_dir}
mkdir -p ${log_dir}
condor_submit job_gpu.sub \
  results_dir=${results_dir} \
  log_dir=${log_dir}