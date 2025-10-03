results_dir=results/${1}
log_dir=logs/${1}
job_script=job_scripts/${1}
command=commands/${1}.txt

mkdir -p ${results_dir}
mkdir -p ${log_dir}

condor_submit job.sub \
  results_dir=${results_dir} \
  log_dir=${log_dir}
  job_script=command
  job_length=${2}