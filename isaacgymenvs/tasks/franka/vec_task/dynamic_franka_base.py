# isaacgymenvs/tasks/franka/vec_task/dynamic_franka_base.py
import torch
from .franka_base import FrankaBaseEnvV2

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
        
        # Call parent step method
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