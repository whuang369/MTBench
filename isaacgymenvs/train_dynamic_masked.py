#!/usr/bin/env python3

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import isaacgymenvs
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.learning.dynamic_mt_a2c_agent import DynamicMTA2CAgent
from isaacgymenvs.tasks.franka.vec_task.dynamic_masked_franka_base import DynamicMaskedFrankaEnvV2
import torch
import time

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    """
    Launch training with dynamic masked multi-task environment.
    """
    # Set seed
    set_seed(cfg.get("seed", 42))
    
    # Create environment with dynamic masking
    print("Creating dynamic masked environment...")
    
    # Override the environment class to use our dynamic masked version
    def create_dynamic_masked_env():
        from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator
        from isaacgymenvs.utils.reformat import omegaconf_to_dict
        
        cfg_dict = omegaconf_to_dict(cfg.task)
        
        # Create environment creator
        create_rlgpu_env = get_rlgames_env_creator(
            seed=cfg.get("seed", 42),
            task_config=cfg_dict,
            task_name=cfg_dict["name"],
            sim_device=cfg.get("sim_device", "cuda:0"),
            rl_device=cfg.get("rl_device", "cuda:0"),
            graphics_device_id=cfg.get("graphics_device_id", 0),
            headless=cfg.get("headless", True),
            multi_gpu=cfg.get("multi_gpu", False),
            virtual_screen_capture=False,
            force_render=True,
        )
        
        # Override the environment class
        original_make = isaacgymenvs.make
        
        def dynamic_masked_make(*args, **kwargs):
            # Create the environment using the original creator
            env = create_rlgpu_env()
            
            # Replace the environment with our dynamic masked version
            if hasattr(env, 'env') and hasattr(env.env, 'cfg'):
                # Create new dynamic masked environment
                dynamic_env = DynamicMaskedFrankaEnvV2(
                    cfg=env.env.cfg,
                    rl_device=env.env.rl_device,
                    sim_device=env.env.sim_device,
                    graphics_device_id=env.env.graphics_device_id,
                    headless=env.env.headless,
                    virtual_screen_capture=env.env.virtual_screen_capture,
                    force_render=env.env.force_render
                )
                
                # Replace the environment
                env.env = dynamic_env
                env.num_envs = dynamic_env.num_envs
                env.single_observation_space = dynamic_env.single_observation_space
                env.single_action_space = dynamic_env.single_action_space
                env.observation_space = dynamic_env.observation_space
                env.action_space = dynamic_env.action_space
            
            return env
        
        # Temporarily replace the make function
        isaacgymenvs.make = dynamic_masked_make
        
        try:
            env = create_rlgpu_env()
        finally:
            # Restore original make function
            isaacgymenvs.make = original_make
        
        return env
    
    # Create environment
    env = create_dynamic_masked_env()
    
    print(f"Created environment with {env.num_envs} total environments")
    print(f"Active environments: {env.get_total_active_environments()}")
    print(f"Task distribution: {env.get_current_task_distribution()}")
    
    # Create agent with dynamic mixture control
    agent_params = {
        'mixture_update_frequency': cfg.task.env.dynamic_masking.get('mixture_update_frequency', 1000),
        'mixture_strategy': cfg.task.env.dynamic_masking.get('mixture_strategy', 'performance_based'),
        'performance_threshold': cfg.task.env.dynamic_masking.get('performance_threshold', 0.5),
        'min_active_envs': cfg.task.env.dynamic_masking.get('min_active_envs', 1),
        'max_active_envs': cfg.task.env.dynamic_masking.get('max_active_envs', 10),
    }
    
    agent = DynamicMTA2CAgent("dynamic_mt_a2c", agent_params)
    
    # Initialize agent with environment
    agent.init(env, cfg.get("seed", 42))
    
    print("Starting training with dynamic masked multi-task environment...")
    print(f"Mixture strategy: {agent.mixture_strategy}")
    print(f"Update frequency: {agent.mixture_update_frequency} steps")
    
    # Training loop
    start_time = time.time()
    last_log_time = start_time
    
    try:
        while True:
            # Play steps and collect experience
            step_time = agent.play_steps()
            
            # Train on collected experience
            train_info = agent.train_epoch()
            
            # Log training information
            current_time = time.time()
            if current_time - last_log_time >= 10:  # Log every 10 seconds
                mixture_stats = agent.get_mixture_statistics()
                print(f"Step {agent.global_steps}:")
                print(f"  Total active environments: {mixture_stats.get('total_active', 'N/A')}")
                print(f"  Task distribution: {env.get_current_task_distribution()}")
                print(f"  Step time: {step_time:.4f}s")
                
                if 'task_0' in mixture_stats:
                    print(f"  Task 0 active ratio: {mixture_stats['task_0']['active_ratio']:.3f}")
                
                last_log_time = current_time
            
            # Check for training completion
            if agent.global_steps >= cfg.get("max_steps", 1000000):
                print("Training completed!")
                break
                
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    # Save final statistics
    performance_history = agent.get_performance_history()
    print(f"Training completed after {agent.global_steps} steps")
    print(f"Final task distribution: {env.get_current_task_distribution()}")
    
    # Save mixture history
    mixture_history = performance_history['mixture_history']
    if mixture_history:
        print(f"Mixture was updated {len(mixture_history)} times")
        print("Final mixture history:")
        for entry in mixture_history[-5:]:  # Show last 5 updates
            print(f"  Step {entry['step']}: {entry['mixture']}")

if __name__ == "__main__":
    launch_rlg_hydra()
