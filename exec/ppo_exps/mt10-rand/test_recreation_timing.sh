#!/bin/bash

# Simple script to test environment recreation timing
# This script runs a very short training session with frequent recreation to measure timing

echo "Environment Recreation Timing Test"
echo "=================================="

# Test parameters - very short run with frequent recreation
num_envs=1024  # Smaller for faster testing
initial_task_counts=[102,102,102,102,102,102,101,101,101,101]
new_task_counts=[80,80,80,80,80,80,120,120,120,120]
recreation_freq=50   # Very frequent recreation for timing measurement
max_iterations=200   # Very short run

echo "Test Configuration:"
echo "  - Number of environments: $num_envs"
echo "  - Initial task counts: $initial_task_counts"
echo "  - New task counts: $new_task_counts"
echo "  - Recreation frequency: $recreation_freq steps"
echo "  - Max iterations: $max_iterations"
echo ""

# Run the test
cmd="python isaacgymenvs/train.py \
    task_id=[4,16,17,18,28,31,38,40,48,49] \
    task_counts=$initial_task_counts \
    num_envs=$num_envs \
    task=meta-world-v2 \
    fixed=False \
    reward_scale=100 \
    termination_on_success=False \
    experiment=recreation_timing_test \
    train=meta-world-mt10-famo-PPO \
    seed=42 \
    wandb_activate=False \
    headless=True \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    record_videos=False \
    max_iterations=$max_iterations \
    update_distribution_steps=$recreation_freq \
    recreate_task_env_count=$new_task_counts"

echo "Running timing test..."
echo "========================================="
echo "Look for timing breakdown messages like:"
echo "  'Environment recreation timing breakdown:'"
echo "  'TOTAL RECREATION TIME: X.XXXX seconds'"
echo "========================================="
echo ""

# Run and capture output
$cmd 2>&1 | grep -E "(Environment recreation|TOTAL RECREATION TIME|recreated successfully)"

echo ""
echo "========================================="
echo "Timing test completed!"
echo "The output above shows the exact recreation times."
echo "========================================="
