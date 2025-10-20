#!/bin/bash

# Test script for environment recreation functionality
# Based on famo.sh but modified to test environment recreation speed

echo "Testing Environment Recreation Speed"
echo "===================================="

# Test parameters
for e in 24576
do
    # Initial task distribution
    initial_task_counts=[2458,2458,2458,2458,2458,2458,2457,2457,2457,2457]
    
    # New task distribution to test recreation
    new_task_counts=[2000,2000,2000,2000,2000,2000,3000,3000,3000,3000]
    
    # Calculate iterations based on environment count
    t=$(( (1000000000 + (e * 32) - 1) / (e * 32) ))
    
    # Test with different recreation frequencies
    for recreation_freq in 1 5000 10000
    do
        echo "Testing with recreation frequency: $recreation_freq steps"
        
        for i in 47 48 49 50 51
        do
            cmd="python isaacgymenvs/train.py \
                task_id=[4,16,17,18,28,31,38,40,48,49] \
                task_counts=$initial_task_counts \
                num_envs=$e \
                task=meta-world-v2 \
                fixed=False \
                reward_scale=100 \
                termination_on_success=False \
                experiment=env_recreation_test_freq_${recreation_freq}_envs_${e}_seed_$i \
                train=meta-world-mt10-famo-PPO \
                seed=$i \
                wandb_activate=True \
                wandb_entity=whuang369-university-of-wisconsin-madison \
                wandb_project=mtbench-env-recreation-tests \
                wandb_group=env-recreation-speed-test \
                headless=True \
                sim_device=cuda:0 \
                rl_device=cuda:0 \
                record_videos=False \
                max_iterations=$t \
                update_distribution_steps=$recreation_freq \
                recreate_task_env_count=$new_task_counts"
            
            echo "Running: $cmd"
            echo "----------------------------------------"
            $cmd
            
            # Add a small delay between runs
            sleep 2
        done
    done
done

echo "Environment recreation speed test completed!"
echo "Check wandb logs for performance metrics and recreation timing."
