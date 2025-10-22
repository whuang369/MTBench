# isaacgymenvs/tasks/franka/vec_task/dynamic_masked_franka_base.py
import torch
import numpy as np
from .franka_base import FrankaBaseEnvV2
from . import task_fns

class DynamicMaskedFrankaEnvV2(FrankaBaseEnvV2):
    """
    Dynamic multi-task environment with masking system for individual environment copies.
    Creates 10x the specified number of environments and uses masking to control which
    environments are active for data collection.
    """
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # Store original configuration
        self.original_cfg = cfg.copy()
        
        # Modify configuration to create 10x environments
        self._modify_config_for_10x_envs(cfg)
        
        # Initialize parent class with modified config
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        # Initialize dynamic masking system
        self._init_dynamic_masking_system()
    
    def _modify_config_for_10x_envs(self, cfg):
        """Modify configuration to create 10x the specified number of environments"""
        # Store original task environment counts
        self.original_task_env_count = cfg["env"]["taskEnvCount"].copy()
        
        # Create 10x environments for each task
        self.extended_task_env_count = [count * 10 for count in self.original_task_env_count]
        
        # Update configuration
        cfg["env"]["taskEnvCount"] = self.extended_task_env_count
        cfg["env"]["numEnvs"] = sum(self.extended_task_env_count)
        
        print(f"Dynamic Masked Environment: Creating 10x environments")
        print(f"Original task env counts: {self.original_task_env_count}")
        print(f"Extended task env counts (10x): {self.extended_task_env_count}")
        print(f"Total environments: {cfg['env']['numEnvs']}")
    
    def _init_dynamic_masking_system(self):
        """Initialize the dynamic masking system"""
        # Environment mask: True = active (collect data), False = inactive (frozen)
        self.env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Track active environments per task
        self.active_envs_per_task = torch.tensor(self.original_task_env_count, device=self.device)
        
        # Initialize with original task distribution
        self._apply_initial_task_distribution()
        
        # Dynamic mixture control
        self.mixture_update_frequency = 1000  # Update every 1000 steps
        self.step_count = 0
        self.last_mixture_update = 0
        
        # Track statistics
        self.task_success_rates = torch.zeros(self.num_tasks, device=self.device)
        self.task_sample_counts = torch.zeros(self.num_tasks, device=self.device)
        
        print(f"Initialized dynamic masking system with {self.num_envs} total environments")
        print(f"Active environments per task: {self.active_envs_per_task}")
    
    def _apply_initial_task_distribution(self):
        """Apply initial task distribution (original counts)"""
        total_count = 0
        for i, (tid, original_count) in enumerate(zip(self.task_idx, self.original_task_env_count)):
            # Deactivate all environments for this task first
            start_idx = total_count
            end_idx = total_count + self.extended_task_env_count[i]
            self.env_mask[start_idx:end_idx] = False
            
            # Activate only the original number of environments
            self.env_mask[start_idx:start_idx + original_count] = True
            
            total_count += self.extended_task_env_count[i]
    
    def step(self, actions):
        """Override step to include dynamic masking and mixture updates"""
        # Update mixture if needed
        self.step_count += 1
        if self.step_count - self.last_mixture_update >= self.mixture_update_frequency:
            self._update_task_mixture()
            self.last_mixture_update = self.step_count
        
        # Get active environment indices
        active_indices = self.env_mask.nonzero().squeeze()
        
        if len(active_indices) == 0:
            # No active environments, return zeros
            return self.obs_dict, torch.zeros_like(self.rew_buf), torch.zeros_like(self.reset_buf), self.extras
        
        # Step only active environments
        self._step_active_environments(active_indices, actions)
        
        # Return full environment state (inactive environments remain frozen)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras
    
    def _step_active_environments(self, active_indices, actions):
        """Step only the active environments while keeping inactive ones frozen"""
        # Store original actions
        original_actions = self.actions.clone()
        
        # Set actions for active environments only
        self.actions = torch.zeros_like(self.actions)
        self.actions[active_indices] = actions[active_indices]
        
        # Apply actions to physics
        self.pre_physics_step(self.actions)
        
        # Step physics simulation
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)
        
        # Fetch results
        self.gym.fetch_results(self.sim, True)
        
        if self.camera_rendering:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
        
        # Compute observations and rewards only for active environments
        self._compute_observations_masked(active_indices)
        self._compute_reward_masked(active_indices)
        
        if self.camera_rendering:
            self.gym.end_access_image_tensors(self.sim)
        
        # Update progress buffer
        self.progress_buf += 1
        
        # Handle resets only for active environments
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        active_reset_ids = env_ids[torch.isin(env_ids, active_indices)]
        if len(active_reset_ids) > 0:
            self.reset_idx(active_reset_ids)
        
        # Update control steps
        self.control_steps += 1
        
        # Update timeouts
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)
        
        # Update extras
        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        
        # Update observation dict
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
        
        # Restore original actions
        self.actions = original_actions
    
    def _compute_observations_masked(self, active_indices):
        """Compute observations only for active environments"""
        # Refresh physics state
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        
        # Compute universal observations for all environments (needed for physics)
        panda_leftfinger_tip_rigid_body_idx = self.franka_rigid_body_start_idx[:] + 10
        panda_rightfinger_tip_rigid_body_idx = self.franka_rigid_body_start_idx[:] + 12
        panda_eef_rigid_body_idx = self.franka_rigid_body_start_idx[:] + 13
        
        panda_leftfinger_tip_rigid_body_states = self.rigid_body_states[panda_leftfinger_tip_rigid_body_idx].view(-1, 13)
        panda_rightfinger_tip_rigid_body_states = self.rigid_body_states[panda_rightfinger_tip_rigid_body_idx].view(-1, 13)
        panda_eef_rigid_body_states = self.rigid_body_states[panda_eef_rigid_body_idx].view(-1, 13)
        
        eef_pos = panda_eef_rigid_body_states[:, 0:3]
        eef_rot = panda_eef_rigid_body_states[:, 3:7]
        
        self.franka_lfinger_pos = panda_leftfinger_tip_rigid_body_states[:, 0:3]
        self.franka_rfinger_pos = panda_rightfinger_tip_rigid_body_states[:, 0:3]
        
        gripper_distance_apart = torch.norm(
            self.franka_rfinger_pos - self.franka_lfinger_pos, dim=-1)
        normalized_openess = torch.clip(
            gripper_distance_apart/.095, 0.0, 1.0).unsqueeze(-1)
        
        if self.tcp_init is None:     
            self.tcp_init = (self.franka_lfinger_pos + self.franka_rfinger_pos) / 2
        
        # Compute task-specific observations only for active environments
        total_count = 0
        env_ids = torch.arange(self.num_envs, device=self.device)
        object_states = []
        
        for tid, env_count in zip(self.task_idx, self.extended_task_env_count):
            task_name = self.task_idx2name[tid]
            obs_fn = getattr(task_fns, task_name).compute_observations
            
            # Only compute for active environments in this task
            task_env_ids = env_ids[total_count:total_count+env_count]
            active_task_env_ids = task_env_ids[torch.isin(task_env_ids, active_indices)]
            
            if len(active_task_env_ids) > 0:
                object_state = obs_fn(self, active_task_env_ids)
                # Create full object state tensor for this task
                full_object_state = torch.zeros(env_count, object_state.shape[1], device=self.device)
                active_mask = torch.isin(task_env_ids, active_indices)
                full_object_state[active_mask] = object_state
                object_states.append(full_object_state)
            else:
                # No active environments for this task, create zeros
                dummy_env_ids = task_env_ids[:1]  # Use first env as dummy
                object_state = obs_fn(self, dummy_env_ids)
                full_object_state = torch.zeros(env_count, object_state.shape[1], device=self.device)
                object_states.append(full_object_state)
            
            total_count += env_count
        
        object_states = torch.cat(object_states, dim=0)
        self.obj_pos = object_states[:, 0:3]
        
        # Combine all observations
        if self.ml_one_enabled:
            self.obs_buf = torch.cat([
                eef_pos,
                normalized_openess, 
                object_states,
                self.obs_buf[:, 0:18],
                torch.zeros_like(self.target_pos)
            ], dim=-1)
        else:
            self.obs_buf = torch.cat([
                eef_pos,
                normalized_openess, 
                object_states,
                self.obs_buf[:, 0:18],
                self.target_pos
            ], dim=-1)
        
        if self.cfg["env"]["taskEmbedding"]:
            self.obs_buf = torch.cat([
                self.obs_buf,
                self.task_embedding
            ], dim=-1)
    
    def _compute_reward_masked(self, active_indices):
        """Compute rewards only for active environments"""
        # Initialize rewards for all environments
        self.rew_buf = torch.zeros_like(self.rew_buf)
        self.reset_buf = torch.zeros_like(self.reset_buf)
        self.success_buf = torch.zeros_like(self.success_buf)
        
        # Compute rewards only for active environments
        total_count = 0
        env_ids = torch.arange(self.num_envs, device=self.device)
        franka_dof_pos = self.dof_state[self.franka_dof_idx, :].view(self.num_envs, -1, 2)[...,0]
        
        for tid, env_count in zip(self.task_idx, self.extended_task_env_count):
            task_name = self.task_idx2name[tid]
            reward_fn = getattr(task_fns, task_name).compute_reward
            
            # Only compute for active environments in this task
            task_env_ids = env_ids[total_count:total_count+env_count]
            active_task_env_ids = task_env_ids[torch.isin(task_env_ids, active_indices)]
            
            if len(active_task_env_ids) > 0:
                # Compute rewards for active environments
                active_rew, active_reset, active_success = reward_fn(
                    self.reset_buf[active_task_env_ids],
                    self.progress_buf[active_task_env_ids], 
                    self.actions[active_task_env_ids], 
                    franka_dof_pos[active_task_env_ids],
                    self.franka_lfinger_pos[active_task_env_ids], 
                    self.franka_rfinger_pos[active_task_env_ids], 
                    self.max_episode_length,
                    self.tcp_init[active_task_env_ids], 
                    self.target_pos[active_task_env_ids], 
                    self.obj_pos[active_task_env_ids], 
                    self.obj_init_pos[active_task_env_ids], 
                    self.specialized_kwargs[task_name][active_task_env_ids[0].item()])
                
                # Store results
                self.rew_buf[active_task_env_ids] = active_rew
                self.reset_buf[active_task_env_ids] = active_reset
                self.success_buf[active_task_env_ids] = active_success
                
                if self.cfg["env"]["sparse_reward"]:
                    self.rew_buf[active_task_env_ids] = active_success.float()
            
            total_count += env_count
        
        if not self.termination_on_success:
            self.reset_buf = (self.progress_buf >= self.max_episode_length - 1)
        
        # Apply reward scaling
        scaling_factor = torch.ones_like(self.rew_buf) * self.reward_scale
        scaling_factor[self.success_buf] = 1.0
        self.rew_buf *= scaling_factor
        
        # Update cumulative rewards
        self.cumulatives["reward"] += self.rew_buf
        self.cumulatives["success"] += self.success_buf
    
    def _update_task_mixture(self):
        """Update the task mixture dynamically"""
        # Calculate current success rates
        self._calculate_task_success_rates()
        
        # Simple strategy: increase sampling for poorly performing tasks
        # and decrease for well-performing tasks
        success_threshold = 0.5
        
        for i in range(self.num_tasks):
            success_rate = self.task_success_rates[i]
            current_active = self.active_envs_per_task[i]
            max_possible = self.extended_task_env_count[i]
            
            if success_rate < success_threshold and current_active < max_possible:
                # Increase active environments for poorly performing tasks
                self.active_envs_per_task[i] = min(current_active + 1, max_possible)
            elif success_rate > success_threshold and current_active > 1:
                # Decrease active environments for well-performing tasks
                self.active_envs_per_task[i] = max(current_active - 1, 1)
        
        # Apply new distribution
        self._apply_task_distribution()
        
        print(f"Updated task mixture - Active envs per task: {self.active_envs_per_task}")
        print(f"Task success rates: {self.task_success_rates}")
    
    def _calculate_task_success_rates(self):
        """Calculate success rates for each task"""
        total_count = 0
        for i, (tid, env_count) in enumerate(zip(self.task_idx, self.extended_task_env_count)):
            task_successes = self.success_buf[total_count:total_count + env_count]
            # Only consider active environments
            active_mask = self.env_mask[total_count:total_count + env_count]
            if active_mask.sum() > 0:
                active_successes = task_successes[active_mask]
                self.task_success_rates[i] = active_successes.float().mean()
            else:
                self.task_success_rates[i] = 0.0
            total_count += env_count
    
    def _apply_task_distribution(self):
        """Apply the current task distribution to environment mask"""
        total_count = 0
        for i, (tid, env_count) in enumerate(zip(self.task_idx, self.extended_task_env_count)):
            start_idx = total_count
            end_idx = total_count + env_count
            
            # Deactivate all environments for this task
            self.env_mask[start_idx:end_idx] = False
            
            # Activate the required number
            active_count = self.active_envs_per_task[i]
            self.env_mask[start_idx:start_idx + active_count] = True
            
            total_count += env_count
    
    def set_task_mixture(self, new_active_counts):
        """Manually set the number of active environments for each task"""
        if len(new_active_counts) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} task counts, got {len(new_active_counts)}")
        
        for i, count in enumerate(new_active_counts):
            max_possible = self.extended_task_env_count[i]
            self.active_envs_per_task[i] = min(max(count, 1), max_possible)
        
        self._apply_task_distribution()
        print(f"Manually set task mixture - Active envs per task: {self.active_envs_per_task}")
    
    def get_current_task_distribution(self):
        """Get current task distribution for logging"""
        return self.active_envs_per_task.cpu().numpy().tolist()
    
    def get_total_active_environments(self):
        """Get total number of active environments"""
        return self.env_mask.sum().item()
    
    def get_task_statistics(self):
        """Get statistics for each task"""
        stats = {}
        total_count = 0
        
        for i, (tid, env_count) in enumerate(zip(self.task_idx, self.extended_task_env_count)):
            task_successes = self.success_buf[total_count:total_count + env_count]
            active_mask = self.env_mask[total_count:total_count + env_count]
            
            stats[f"task_{tid}"] = {
                "active_envs": active_mask.sum().item(),
                "total_envs": env_count,
                "success_rate": task_successes[active_mask].float().mean().item() if active_mask.sum() > 0 else 0.0,
                "active_ratio": active_mask.float().mean().item()
            }
            total_count += env_count
        
        return stats
