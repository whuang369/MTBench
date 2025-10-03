# isaacgymenvs/tasks/franka/vec_task/dynamic_franka_base.py
import torch
from .franka_base import FrankaBaseEnvV2
from . import task_fns

class DynamicTaskFrankaEnvV2(FrankaBaseEnvV2):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # Initialize parent class
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        # Dynamic task distribution settings
        self.dynamic_config = cfg["env"].get("dynamic_task_distribution", {})
        self.dynamic_enabled = self.dynamic_config.get("enabled", False)
        
        if self.dynamic_enabled:
            self._init_dynamic_task_system()
    
    def _init_dynamic_task_system(self):
        """Initialize dynamic task distribution system"""
        self.dynamic_type = self.dynamic_config.get("type", "performance_based")
        self.update_frequency = self.dynamic_config.get("update_frequency", 1000)
        self.step_count = 0
        
        # Initialize task tracking
        self.task_success_rates = torch.zeros(self.num_tasks, device=self.device)
        self.task_sample_counts = torch.zeros(self.num_tasks, device=self.device)
        self.task_weights = torch.ones(self.num_tasks, device=self.device)
        
        # Environment masking for efficient dynamic allocation
        self.env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.original_task_counts = self.task_env_count.copy()
        
        if self.dynamic_type == "performance_based":
            self._init_performance_based()
        elif self.dynamic_type == "schedule_based":
            self._init_schedule_based()
        elif self.dynamic_type == "curriculum":
            self._init_curriculum()
    
    def _init_performance_based(self):
        """Initialize performance-based dynamic allocation"""
        config = self.dynamic_config.get("performance_based", {})
        self.success_threshold = config.get("success_rate_threshold", 0.8)
        self.min_weight = config.get("min_task_weight", 0.1)
        self.max_weight = config.get("max_task_weight", 5.0)
        self.adaptation_rate = config.get("adaptation_rate", 0.1)
    
    def _init_schedule_based(self):
        """Initialize schedule-based dynamic allocation"""
        self.schedule = self.dynamic_config.get("schedule_based", {}).get("schedule", {})
        self.current_schedule_step = 0
    
    def _init_curriculum(self):
        """Initialize curriculum-based dynamic allocation"""
        config = self.dynamic_config.get("curriculum", {})
        self.easy_tasks = config.get("easy_tasks", [0, 1, 2, 3, 4, 5])
        self.hard_tasks = config.get("hard_tasks", [6, 7, 8, 9])
        self.initial_easy_weight = config.get("initial_easy_weight", 1.0)
        self.initial_hard_weight = config.get("initial_hard_weight", 0.5)
        self.final_easy_weight = config.get("final_easy_weight", 0.2)
        self.final_hard_weight = config.get("final_hard_weight", 1.0)
        self.transition_steps = config.get("transition_steps", 10000)
    
    def step(self, actions):
        """Override step to include dynamic task updates"""
        if self.dynamic_enabled:
            self.step_count += 1
            if self.step_count % self.update_frequency == 0:
                self._update_task_distribution()
            
            # Filter actions to only active environments
            active_indices = self.env_mask.nonzero().squeeze()
            if len(active_indices) == 0:
                # No active environments, return zeros
                return self.obs_dict, torch.zeros_like(self.rew_buf), torch.zeros_like(self.reset_buf), self.extras
            
            active_actions = actions[active_indices]
            
            # Step only active environments using parent's step method
            # We need to temporarily modify the environment state for stepping
            self._step_active_environments(active_indices, active_actions)
            
            # Return the full environment state (with inactive environments frozen)
            return self.obs_dict, self.rew_buf, self.reset_buf, self.extras
        else:
            return super().step(actions)
    
    def _update_task_distribution(self):
        """Update task distribution based on current strategy"""
        if self.dynamic_type == "performance_based":
            self._update_performance_based()
        elif self.dynamic_type == "schedule_based":
            self._update_schedule_based()
        elif self.dynamic_type == "curriculum":
            self._update_curriculum()
    
    def _update_performance_based(self):
        """Update based on task performance"""
        self._calculate_task_success_rates()
        
        # Adjust weights based on performance
        for i in range(self.num_tasks):
            success_rate = self.task_success_rates[i]
            if success_rate > self.success_threshold:
                # Reduce weight for well-performing tasks
                self.task_weights[i] *= (1 - self.adaptation_rate)
            else:
                # Increase weight for poorly-performing tasks
                self.task_weights[i] *= (1 + self.adaptation_rate)
            
            # Clamp weights
            self.task_weights[i] = torch.clamp(self.task_weights[i], self.min_weight, self.max_weight)
        
        self._apply_task_weights()
    
    def _update_schedule_based(self):
        """Update based on predefined schedule"""
        # Find the appropriate schedule step
        for step in sorted(self.schedule.keys(), reverse=True):
            if self.step_count >= step:
                new_counts = self.schedule[step]
                self._apply_new_task_counts(new_counts)
                break
    
    def _update_curriculum(self):
        """Update based on curriculum learning"""
        progress = min(self.step_count / self.transition_steps, 1.0)
        
        # Interpolate between initial and final weights
        easy_weight = self.initial_easy_weight + (self.final_easy_weight - self.initial_easy_weight) * progress
        hard_weight = self.initial_hard_weight + (self.final_hard_weight - self.initial_hard_weight) * progress
        
        # Apply weights to tasks
        for i, tid in enumerate(self.task_idx):
            if tid in self.easy_tasks:
                self.task_weights[i] = easy_weight
            elif tid in self.hard_tasks:
                self.task_weights[i] = hard_weight
        
        self._apply_task_weights()
    
    def _calculate_task_success_rates(self):
        """Calculate success rates for each task"""
        total_count = 0
        for i, (tid, count) in enumerate(zip(self.task_idx, self.task_env_count)):
            task_successes = self.success_buf[total_count:total_count + count]
            self.task_success_rates[i] = task_successes.float().mean()
            total_count += count
    
    def _apply_task_weights(self):
        """Apply current task weights to environment mask"""
        # Calculate new task counts based on weights
        total_weight = self.task_weights.sum()
        new_counts = (self.task_weights / total_weight * self.num_envs).int()
        
        # Ensure we don't exceed original counts
        for i in range(len(new_counts)):
            new_counts[i] = min(new_counts[i], self.original_task_counts[i])
        
        self._apply_new_task_counts(new_counts.tolist())
    
    def _apply_new_task_counts(self, new_counts):
        """Apply new task counts by updating environment mask"""
        total_count = 0
        for i, (tid, new_count) in enumerate(zip(self.task_idx, new_counts)):
            start_idx = total_count
            end_idx = total_count + self.original_task_counts[i]
            
            # Deactivate all environments for this task
            self.env_mask[start_idx:end_idx] = False
            
            # Reactivate the required number
            self.env_mask[start_idx:start_idx + new_count] = True
            
            total_count += self.original_task_counts[i]
    
    def get_current_task_distribution(self):
        """Get current task distribution for logging"""
        if not self.dynamic_enabled:
            return self.task_env_count
        
        distribution = []
        total_count = 0
        for i, (tid, original_count) in enumerate(zip(self.task_idx, self.original_task_counts)):
            active_count = self.env_mask[total_count:total_count + original_count].sum().item()
            distribution.append(active_count)
            total_count += original_count
        
        return distribution
    
    def _step_active_environments(self, active_indices, active_actions):
        """Step only the active environments while keeping inactive ones frozen"""
        # Store original actions
        original_actions = self.actions.clone()
        
        # Set actions for active environments only
        self.actions = torch.zeros_like(self.actions)
        self.actions[active_indices] = active_actions
        
        # Call parent's step method but override compute_reward and compute_observations
        # to only process active environments
        self._step_with_masking(active_indices)
        
        # Restore original actions
        self.actions = original_actions
    
    def _step_with_masking(self, active_indices):
        """Step the environment with masking applied"""
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
        
        for tid, env_count in zip(self.task_idx, self.task_env_count):
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
        
        for tid, env_count in zip(self.task_idx, self.task_env_count):
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