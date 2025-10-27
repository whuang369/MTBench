results_dir=results/${1}
log_dir=logs/${1}
commands_file=commands/${1}

mkdir -p ${results_dir}
mkdir -p ${log_dir}

condor_submit job.sub \
  results_dir=${results_dir} \
  log_dir=${log_dir} \
  commands_file=${commands_file} \
  mem=${2} \
  disk=${3} \
  gpu_mem=${4} \
  job_length=${5}