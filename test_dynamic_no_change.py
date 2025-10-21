#!/usr/bin/env python3
"""
Test script for dynamic task environment counts system - NO DISTRIBUTION CHANGE
This script runs with dynamic masking but doesn't change the distribution,
producing the same result as famo.sh but with 10x environments and masking.
"""

import os
import sys
import torch
import time
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import isaacgymenvs
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.learning.dynamic_mt_a2c_agent import DynamicMTA2CAgent
from isaacgymenvs.tasks.franka.vec_task.dynamic_masked_franka_base import DynamicMaskedFrankaEnvV2

def create_dynamic_masked_env_with_no_change():
    """Create dynamic masked environment that doesn't change distribution"""
    
    # Configuration matching famo.sh parameters
    cfg = {
        "task_id": [4, 16, 17, 18, 28, 31, 38, 40, 48, 49],
        "task_counts": [2458, 2458, 2458, 2458, 2458, 2458, 2457, 2457, 2457, 2457],
        "num_envs": 24576,
        "task": "meta-world-v2",
        "fixed": False,
        "reward_scale": 100,
        "termination_on_success": False,
        "seed": 47,
        "headless": True,
        "sim_device": "cuda:0",
        "rl_device": "cuda:0",
        "record_videos": False,
        "max_iterations": 1000,  # Reduced for testing
        "dynamic_masking": {
            "enabled": True,
            "mixture_update_frequency": 1000000,  # Very high - effectively no updates
            "mixture_strategy": "uniform",  # Keep uniform distribution
            "performance_threshold": 0.5,
            "min_active_envs": 1,
            "max_active_envs": 10
        }
    }
    
    # Create environment configuration
    env_cfg = {
        "env": {
            "numEnvs": cfg["num_envs"],
            "taskEnvCount": cfg["task_counts"],
            "tasks": cfg["task_id"],
            "episodeLength": 150,
            "envSpacing": 1.5,
            "clipObservations": 5.0,
            "clipActions": 1.0,
            "actionScale": 0.01,
            "resetNoise": 0.15,
            "taskEmbedding": True,
            "sparse_reward": False,
            "termination_on_success": cfg["termination_on_success"],
            "reward_scale": cfg["reward_scale"],
            "init_at_random_progress": False,
            "exemptedInitAtRandomProgressTasks": [],
            "dynamic_masking": cfg["dynamic_masking"]
        },
        "sim": {
            "dt": 0.01667,
            "substeps": 2,
            "up_axis": "z",
            "use_gpu_pipeline": True,
            "gravity": [0.0, 0.0, -9.81]
        }
    }
    
    # Create dynamic masked environment
    env = DynamicMaskedFrankaEnvV2(
        cfg=env_cfg,
        rl_device=cfg["rl_device"],
        sim_device=cfg["sim_device"],
        graphics_device_id=0,
        headless=cfg["headless"],
        virtual_screen_capture=False,
        force_render=False
    )
    
    return env, cfg

def run_training_no_change():
    """Run training with dynamic masking but no distribution changes"""
    
    print("=" * 80)
    print("DYNAMIC MASKED ENVIRONMENT - NO DISTRIBUTION CHANGE")
    print("=" * 80)
    print("This test runs with 10x environments and masking but keeps the same")
    print("distribution as famo.sh (no dynamic changes)")
    print("=" * 80)
    
    # Set seed
    set_seed(47)
    
    # Create environment
    print("Creating dynamic masked environment...")
    env, cfg = create_dynamic_masked_env_with_no_change()
    
    print(f"✓ Environment created successfully")
    print(f"  Original task counts: {env.original_task_env_count}")
    print(f"  Extended task counts: {env.extended_task_env_count}")
    print(f"  Total environments: {env.num_envs}")
    print(f"  Active environments: {env.get_total_active_environments()}")
    print(f"  Initial distribution: {env.get_current_task_distribution()}")
    
    # Create agent with no-change strategy
    agent_params = {
        'mixture_strategy': 'uniform',  # Keep uniform distribution
        'mixture_update_frequency': 1000000,  # Very high - no updates
        'performance_threshold': 0.5,
        'min_active_envs': 1,
        'max_active_envs': 10
    }
    
    agent = DynamicMTA2CAgent("no_change_agent", agent_params)
    agent.init(env, seed=cfg["seed"])
    
    print(f"✓ Agent created with no-change strategy")
    print(f"  Mixture strategy: {agent.mixture_strategy}")
    print(f"  Update frequency: {agent.mixture_update_frequency} steps")
    
    # Training loop
    print("\nStarting training (no distribution changes)...")
    start_time = time.time()
    last_log_time = start_time
    
    try:
        for epoch in range(cfg["max_iterations"]):
            # Play steps
            step_time = agent.play_steps()
            
            # Log every 10 seconds
            current_time = time.time()
            if current_time - last_log_time >= 10:
                current_dist = env.get_current_task_distribution()
                active_envs = env.get_total_active_environments()
                
                print(f"Epoch {epoch+1}:")
                print(f"  Distribution: {current_dist}")
                print(f"  Active environments: {active_envs}")
                print(f"  Step time: {step_time:.4f}s")
                print(f"  Global steps: {agent.global_steps}")
                
                # Verify distribution hasn't changed
                if current_dist != env.original_task_env_count:
                    print(f"  ⚠️  WARNING: Distribution changed from original!")
                else:
                    print(f"  ✓ Distribution unchanged (as expected)")
                
                last_log_time = current_time
            
            # Check for completion
            if agent.global_steps >= cfg["max_iterations"] * 32:  # 32 steps per epoch
                break
                
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    # Final statistics
    final_time = time.time() - start_time
    final_dist = env.get_current_task_distribution()
    
    print(f"\n" + "=" * 80)
    print("TRAINING COMPLETED - NO DISTRIBUTION CHANGE")
    print("=" * 80)
    print(f"Total training time: {final_time:.2f} seconds")
    print(f"Final distribution: {final_dist}")
    print(f"Original distribution: {env.original_task_env_count}")
    print(f"Distribution changed: {final_dist != env.original_task_env_count}")
    print(f"Total active environments: {env.get_total_active_environments()}")
    print(f"Total environments: {env.num_envs}")
    
    # Verify no changes occurred
    if final_dist == env.original_task_env_count:
        print("✓ SUCCESS: Distribution remained unchanged (matches famo.sh behavior)")
    else:
        print("✗ FAILURE: Distribution changed unexpectedly")
    
    env.close()
    return final_dist == env.original_task_env_count

if __name__ == "__main__":
    success = run_training_no_change()
    if success:
        print("\n🎉 Test passed: Dynamic masking works without changing distribution")
    else:
        print("\n❌ Test failed: Distribution changed unexpectedly")
        sys.exit(1)
