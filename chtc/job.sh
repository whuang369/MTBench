#!/bin/bash

cmd=`$1`
echo $cmd

git clone https://github.com/whuang369/MTBench
cd MTBench
git checkout dynamic_task_counts
source install.sh
wandb login d0dbec0b8f79cdb57ef36ae46bb16a336954b2ca

$cmd

tar -zcvf runs.tar.gz runs
tar -zcvf wandb.tar.gz wandb
cp runs.tar.gz /staging/whuang369/MTBench_Results/
cp wandb.tar.gz /staging/whuang369/MTBench_Results/
mv runs.tar.gz ..
mv wandb.tar.gz ..