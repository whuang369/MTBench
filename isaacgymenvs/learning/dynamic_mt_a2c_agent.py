# isaacgymenvs/learning/dynamic_mt_a2c_agent.py
import torch
import time
from .mt_a2c_agent import MTA2CAgent
from . import a2c_common

class DynamicMTA2CAgent(MTA2CAgent):
    """
    Multi-task A2C agent with dynamic environment masking and mixture control.
    Supports dynamic adjustment of task mixture during training.
    """
    
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        
        # Dynamic mixture control parameters
        self.mixture_update_frequency = params.get('mixture_update_frequency', 1000)
        self.last_mixture_update = 0
        self.mixture_update_steps = params.get('mixture_update_steps', 1000)
        
        # Task mixture strategies
        self.mixture_strategy = params.get('mixture_strategy', 'performance_based')
        self.performance_threshold = params.get('performance_threshold', 0.5)
        self.min_active_envs = params.get('min_active_envs', 1)
        self.max_active_envs = params.get('max_active_envs', 10)
        
        # Statistics tracking
        self.task_performance_history = []
        self.mixture_history = []
        
        print(f"Initialized DynamicMTA2CAgent with mixture strategy: {self.mixture_strategy}")
        print(f"Mixture update frequency: {self.mixture_update_frequency} steps")
    
    def play_steps(self):
        """Override play_steps to include dynamic mixture updates"""
        update_list = self.update_list
        
        step_time = 0.0
        
        # Check if we need to update task mixture
        if (self.mixture_update_steps is not None and 
            self.global_steps - self.last_mixture_update >= self.mixture_update_steps):
            self._update_task_mixture()
            self.last_mixture_update = self.global_steps
        
        for n in range(self.horizon_length):
            self.global_steps += 1
            
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])
            
            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()
            
            step_time += (step_time_end - step_time_start)
            
            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()
            
            if self.normalize_reward:
                self.reward_normalizer.update_stats(
                    shaped_rewards.squeeze(-1),
                    self.dones,
                    self.all_task_indices
                )
            
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            
            self.current_rewards += rewards
            self.current_lengths += 1
            
            # Track task performance
            self._track_task_performance(infos)
            
            all_done_indices = self.dones.nonzero(as_tuple=False)
            if len(all_done_indices) > 0:
                self._process_done_indices(all_done_indices, infos)
            
            if self.has_central_value:
                self.experience_buffer.update_data('next_values', n, self._eval_critic(self.obs))
            else:
                self.experience_buffer.update_data('next_values', n, res_dict['values'])
        
        return step_time
    
    def _update_task_mixture(self):
        """Update the task mixture based on current strategy"""
        if not hasattr(self.vec_env, 'set_task_mixture'):
            print("Environment does not support dynamic task mixture")
            return
        
        if self.mixture_strategy == 'performance_based':
            self._update_performance_based_mixture()
        elif self.mixture_strategy == 'uniform':
            self._update_uniform_mixture()
        elif self.mixture_strategy == 'curriculum':
            self._update_curriculum_mixture()
        elif self.mixture_strategy == 'random':
            self._update_random_mixture()
        else:
            print(f"Unknown mixture strategy: {self.mixture_strategy}")
            return
        
        # Log current mixture
        current_mixture = self.vec_env.get_current_task_distribution()
        self.mixture_history.append({
            'step': self.global_steps,
            'mixture': current_mixture.copy(),
            'strategy': self.mixture_strategy
        })
        
        print(f"Updated task mixture at step {self.global_steps}: {current_mixture}")
    
    def _update_performance_based_mixture(self):
        """Update mixture based on task performance"""
        if not hasattr(self.vec_env, 'get_task_statistics'):
            print("Environment does not support task statistics")
            return
        
        stats = self.vec_env.get_task_statistics()
        new_active_counts = []
        
        for i, task_idx in enumerate(self.vec_env.task_idx):
            task_key = f"task_{task_idx}"
            if task_key in stats:
                success_rate = stats[task_key]['success_rate']
                current_active = stats[task_key]['active_envs']
                max_possible = stats[task_key]['total_envs']
                
                # Adjust based on performance
                if success_rate < self.performance_threshold and current_active < max_possible:
                    # Increase active environments for poorly performing tasks
                    new_count = min(current_active + 1, max_possible)
                elif success_rate > self.performance_threshold and current_active > self.min_active_envs:
                    # Decrease active environments for well-performing tasks
                    new_count = max(current_active - 1, self.min_active_envs)
                else:
                    new_count = current_active
                
                new_active_counts.append(new_count)
            else:
                new_active_counts.append(self.min_active_envs)
        
        self.vec_env.set_task_mixture(new_active_counts)
    
    def _update_uniform_mixture(self):
        """Update to uniform mixture across all tasks"""
        if not hasattr(self.vec_env, 'extended_task_env_count'):
            print("Environment does not support extended task counts")
            return
        
        # Calculate uniform distribution
        total_active = self.vec_env.get_total_active_environments()
        num_tasks = len(self.vec_env.task_idx)
        uniform_count = max(total_active // num_tasks, self.min_active_envs)
        
        new_active_counts = [uniform_count] * num_tasks
        self.vec_env.set_task_mixture(new_active_counts)
    
    def _update_curriculum_mixture(self):
        """Update mixture based on curriculum learning"""
        if not hasattr(self.vec_env, 'extended_task_env_count'):
            print("Environment does not support extended task counts")
            return
        
        # Simple curriculum: start with easy tasks, gradually include hard tasks
        progress = min(self.global_steps / 50000, 1.0)  # 50k steps curriculum
        
        new_active_counts = []
        for i, task_idx in enumerate(self.vec_env.task_idx):
            max_possible = self.vec_env.extended_task_env_count[i]
            
            # Easy tasks (0-4) get more environments early, hard tasks (5-9) get more later
            if task_idx < 5:  # Easy tasks
                weight = 1.0 - progress * 0.5  # Decrease over time
            else:  # Hard tasks
                weight = 0.5 + progress * 0.5  # Increase over time
            
            new_count = max(int(weight * max_possible), self.min_active_envs)
            new_count = min(new_count, max_possible)
            new_active_counts.append(new_count)
        
        self.vec_env.set_task_mixture(new_active_counts)
    
    def _update_random_mixture(self):
        """Update mixture randomly"""
        if not hasattr(self.vec_env, 'extended_task_env_count'):
            print("Environment does not support extended task counts")
            return
        
        new_active_counts = []
        for i, task_idx in enumerate(self.vec_env.task_idx):
            max_possible = self.vec_env.extended_task_env_count[i]
            # Randomly choose between min and max
            new_count = torch.randint(
                self.min_active_envs, 
                max_possible + 1, 
                (1,)
            ).item()
            new_active_counts.append(new_count)
        
        self.vec_env.set_task_mixture(new_active_counts)
    
    def _track_task_performance(self, infos):
        """Track performance metrics for each task"""
        if 'final_info' in infos:
            for info in infos['final_info']:
                if info and 'episode' in info:
                    # Extract task information if available
                    task_id = info.get('task_id', 0)
                    success = info.get('success', False)
                    reward = info['episode']['r']
                    
                    self.task_performance_history.append({
                        'step': self.global_steps,
                        'task_id': task_id,
                        'success': success,
                        'reward': reward
                    })
    
    def get_mixture_statistics(self):
        """Get statistics about the current task mixture"""
        if not hasattr(self.vec_env, 'get_task_statistics'):
            return {}
        
        stats = self.vec_env.get_task_statistics()
        stats['total_active'] = self.vec_env.get_total_active_environments()
        stats['mixture_strategy'] = self.mixture_strategy
        stats['update_frequency'] = self.mixture_update_frequency
        
        return stats
    
    def set_mixture_strategy(self, strategy):
        """Change the mixture strategy during training"""
        if strategy in ['performance_based', 'uniform', 'curriculum', 'random']:
            self.mixture_strategy = strategy
            print(f"Changed mixture strategy to: {strategy}")
        else:
            print(f"Unknown mixture strategy: {strategy}")
    
    def set_mixture_update_frequency(self, frequency):
        """Change the mixture update frequency"""
        self.mixture_update_frequency = frequency
        self.mixture_update_steps = frequency
        print(f"Changed mixture update frequency to: {frequency} steps")
    
    def get_performance_history(self):
        """Get performance history for analysis"""
        return {
            'task_performance': self.task_performance_history,
            'mixture_history': self.mixture_history
        }
