#!/bin/bash
for e in 24576
do
	task_counts=[24576]
	t=$(( (10000000 + (e * 32) - 1) / (e * 32) ))
  cmd="python isaacgymenvs/train.py \
    task_id=[16] \
    task_counts=$task_counts \
    num_envs=$e \
    task=meta-world-v2 \
    fixed=False \
    reward_scale=100 \
    termination_on_success=False \
    experiment=05_09_ppo_famo_mt10_rand_envs_${e}_seed_$i \
    train=meta-world-mt10-famo-PPO \
    seed=47 \
    wandb_activate=True \
    wandb_entity=whuang369-university-of-wisconsin-madison \
    wandb_project=mtbench-test-mask \
    wandb_group=mt10-famo \
    headless=True \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    record_videos=False \
    max_iterations=$t \
    enable_mask=False"
  echo $cmd
  $cmd
done

