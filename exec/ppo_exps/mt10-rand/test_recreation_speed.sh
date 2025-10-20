#!/bin/bash

# Simple test script to measure environment recreation speed
# This script runs a short training session with frequent environment recreation

echo "Environment Recreation Speed Test"
echo "================================="

# Test parameters
num_envs=4096
initial_task_counts=[410,410,410,410,410,410,409,409,409,409]
new_task_counts=[300,300,300,300,300,300,600,600,600,600]
recreation_freq=100  # Very frequent recreation for speed testing
max_iterations=1000  # Short run for speed testing

echo "Initial task counts: $initial_task_counts"
echo "New task counts: $new_task_counts"
echo "Recreation frequency: $recreation_freq steps"
echo "Max iterations: $max_iterations"
echo ""

for seed in 42 43 44
do
    echo "Running test with seed: $seed"
    echo "----------------------------------------"
    
    cmd="python isaacgymenvs/train.py \
        task_id=[4,16,17,18,28,31,38,40,48,49] \
        task_counts=$initial_task_counts \
        num_envs=$num_envs \
        task=meta-world-v2 \
        fixed=False \
        reward_scale=100 \
        termination_on_success=False \
        experiment=recreation_speed_test_seed_$seed \
        train=meta-world-mt10-famo-PPO \
        seed=$seed \
        wandb_activate=True \
        wandb_entity=whuang369-university-of-wisconsin-madison \
        wandb_project=mtbench-recreation-speed \
        wandb_group=speed-test \
        headless=True \
        sim_device=cuda:0 \
        rl_device=cuda:0 \
        record_videos=False \
        max_iterations=$max_iterations \
        update_distribution_steps=$recreation_freq \
        recreate_task_env_count=$new_task_counts"
    
    echo "Command: $cmd"
    echo ""
    
    # Run the command and capture timing
    start_time=$(date +%s)
    echo "Starting training with environment recreation..."
    echo "Look for 'Environment recreation timing breakdown:' messages in the output below:"
    echo "========================================="
    $cmd
    end_time=$(date +%s)
    
    duration=$((end_time - start_time))
    echo "========================================="
    echo "Test completed in $duration seconds"
    echo "Check the output above for detailed recreation timing breakdown"
    echo "========================================="
    echo ""
done

echo "All speed tests completed!"
echo "Check the console output for recreation timing messages."
echo "Check wandb for detailed performance metrics."
