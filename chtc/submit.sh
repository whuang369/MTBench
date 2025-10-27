results_dir=results/${1}
log_dir=logs/${1}
commands_file=commands/${1}.txt

mkdir -p ${results_dir}
mkdir -p ${log_dir}

condor_submit job.sub \
  results_dir=${results_dir} \
  log_dir=${log_dir} \
  commands_file=${commands_file} \
  job_length=${2} \
  gpu_mem=${3}