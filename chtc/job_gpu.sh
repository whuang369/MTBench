#!/bin/bash
git clone https://github.com/whuang369/MTBench
cd MTBench
git checkout dynamic_task_counts
source install.sh
export WANDB_MODE=offline
sh exec/ppo_exps/mt10-rand/famo.sh
tar -zcvf runs.tar.gz runs
tar -zcvf wandb.tar.gz wandb
cp runs.tar.gz /staging/whuang369/MTBench_Results/
cp wandb.tar.gz /staging/whuang369/MTBench_Results/
mv runs.tar.gz ..
mv wandb.tar.gz ..