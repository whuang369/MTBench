#!/usr/bin/env python3
"""
Test script for dynamic task environment counts system - SINGLE DISTRIBUTION CHANGE
This script runs with dynamic masking and changes the distribution once in the middle
of training, demonstrating the dynamic mixture control functionality.
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

def create_dynamic_masked_env_with_single_change():
    """Create dynamic masked environment that changes distribution once"""
    
    # Configuration matching famo.sh parameters but smaller for testing
    cfg = {
        "task_id": [4, 16, 17, 18, 28, 31, 38, 40, 48, 49],
        "task_counts": [245, 245, 245, 245, 245, 245, 245, 245, 245, 245],  # Reduced for testing
        "num_envs": 2450,  # Reduced for testing
        "task": "meta-world-v2",
        "fixed": False,
        "reward_scale": 100,
        "termination_on_success": False,
        "seed": 47,
        "headless": True,
        "sim_device": "cuda:0",
        "rl_device": "cuda:0",
        "record_videos": False,
        "max_iterations": 200,  # Reduced for testing
        "change_epoch": 100,  # Change distribution at epoch 100
        "new_distribution": [100, 100, 100, 100, 100, 50, 50, 50, 50, 50],  # New distribution
        "dynamic_masking": {
            "enabled": True,
            "mixture_update_frequency": 1000000,  # Very high - manual control only
            "mixture_strategy": "manual",  # Manual control
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

def run_training_single_change():
    """Run training with dynamic masking and single distribution change"""
    
    print("=" * 80)
    print("DYNAMIC MASKED ENVIRONMENT - SINGLE DISTRIBUTION CHANGE")
    print("=" * 80)
    print("This test runs with 10x environments and masking, then changes")
    print("the distribution once in the middle of training")
    print("=" * 80)
    
    # Set seed
    set_seed(47)
    
    # Create environment
    print("Creating dynamic masked environment...")
    env, cfg = create_dynamic_masked_env_with_single_change()
    
    print(f"✓ Environment created successfully")
    print(f"  Original task counts: {env.original_task_env_count}")
    print(f"  Extended task counts: {env.extended_task_env_count}")
    print(f"  Total environments: {env.num_envs}")
    print(f"  Active environments: {env.get_total_active_environments()}")
    print(f"  Initial distribution: {env.get_current_task_distribution()}")
    print(f"  Change scheduled at epoch: {cfg['change_epoch']}")
    print(f"  New distribution: {cfg['new_distribution']}")
    
    # Create agent with manual control
    agent_params = {
        'mixture_strategy': 'uniform',  # Start with uniform
        'mixture_update_frequency': 1000000,  # Very high - manual control only
        'performance_threshold': 0.5,
        'min_active_envs': 1,
        'max_active_envs': 10
    }
    
    agent = DynamicMTA2CAgent("single_change_agent", agent_params)
    agent.init(env, seed=cfg["seed"])
    
    print(f"✓ Agent created with manual control")
    print(f"  Mixture strategy: {agent.mixture_strategy}")
    print(f"  Update frequency: {agent.mixture_update_frequency} steps")
    
    # Training loop
    print(f"\nStarting training (distribution change at epoch {cfg['change_epoch']})...")
    start_time = time.time()
    last_log_time = start_time
    distribution_changed = False
    
    try:
        for epoch in range(cfg["max_iterations"]):
            # Check if it's time to change distribution
            if epoch == cfg["change_epoch"] and not distribution_changed:
                print(f"\n🔄 CHANGING DISTRIBUTION AT EPOCH {epoch}")
                print(f"  Old distribution: {env.get_current_task_distribution()}")
                print(f"  New distribution: {cfg['new_distribution']}")
                
                # Change distribution
                env.set_task_mixture(cfg["new_distribution"])
                distribution_changed = True
                
                print(f"  ✓ Distribution changed successfully")
                print(f"  New active environments: {env.get_total_active_environments()}")
                print(f"  New distribution: {env.get_current_task_distribution()}")
            
            # Play steps
            step_time = agent.play_steps()
            
            # Log every 20 seconds or at change point
            current_time = time.time()
            should_log = (current_time - last_log_time >= 20) or (epoch == cfg["change_epoch"])
            
            if should_log:
                current_dist = env.get_current_task_distribution()
                active_envs = env.get_total_active_environments()
                
                print(f"Epoch {epoch+1}:")
                print(f"  Distribution: {current_dist}")
                print(f"  Active environments: {active_envs}")
                print(f"  Step time: {step_time:.4f}s")
                print(f"  Global steps: {agent.global_steps}")
                
                if epoch == cfg["change_epoch"]:
                    print(f"  🎯 DISTRIBUTION CHANGE COMPLETED")
                elif epoch < cfg["change_epoch"]:
                    print(f"  ⏳ Waiting for distribution change...")
                else:
                    print(f"  ✅ Post-change training...")
                
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
    print("TRAINING COMPLETED - SINGLE DISTRIBUTION CHANGE")
    print("=" * 80)
    print(f"Total training time: {final_time:.2f} seconds")
    print(f"Final distribution: {final_dist}")
    print(f"Original distribution: {env.original_task_env_count}")
    print(f"Target distribution: {cfg['new_distribution']}")
    print(f"Distribution changed: {distribution_changed}")
    print(f"Total active environments: {env.get_total_active_environments()}")
    print(f"Total environments: {env.num_envs}")
    
    # Verify change occurred
    if distribution_changed and final_dist == cfg['new_distribution']:
        print("✓ SUCCESS: Distribution changed successfully as planned")
        return True
    elif not distribution_changed:
        print("✗ FAILURE: Distribution change was not triggered")
        return False
    else:
        print("✗ FAILURE: Distribution change occurred but result doesn't match target")
        return False

if __name__ == "__main__":
    success = run_training_single_change()
    if success:
        print("\n🎉 Test passed: Dynamic masking with single distribution change works")
    else:
        print("\n❌ Test failed: Single distribution change did not work as expected")
        sys.exit(1)
