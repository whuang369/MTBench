#!/usr/bin/env python3

import sys
import os
import time
from datetime import datetime

# noinspection PyUnresolvedReferences
import isaacgym

import hydra
from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
import gym

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.learning.dynamic_mt_a2c_agent import DynamicMTA2CAgent
from isaacgymenvs.tasks.franka.vec_task.dynamic_masked_franka_base import DynamicMaskedFrankaEnvV2

def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')
    
    # Add dynamic masking parameters
    if hasattr(cfg, 'mixture_update_frequency') and cfg.mixture_update_frequency is not None:
        train_cfg['mixture_update_frequency'] = cfg.mixture_update_frequency
        print(f'Added mixture_update_frequency: {cfg.mixture_update_frequency}')
    
    if hasattr(cfg, 'mixture_strategy') and cfg.mixture_strategy is not None:
        train_cfg['mixture_strategy'] = cfg.mixture_strategy
        print(f'Added mixture_strategy: {cfg.mixture_strategy}')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv, VisualRLGPUAlgoObserver
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import mt_a2c_agent, ml_a2c_agent, mt_models, mt_player
    from isaacgymenvs.learning.networks import pq_builder, soft_modularized_pq_builder
    from isaacgymenvs.learning import mt_sac_agent, mt_sac_agent_gradmani
    from isaacgymenvs.learning import pqn_agent
    from isaacgymenvs.learning import grpo_agent
    from isaacgymenvs.learning import td3_agent 

    from isaacgymenvs.learning.networks import soft_modularization_sac_builder
    from isaacgymenvs.learning.networks import soft_modularization_a2c_builder
    from isaacgymenvs.learning.networks import moore_sac_builder
    from isaacgymenvs.learning.networks import moore_a2c_builder
    from isaacgymenvs.learning.networks import care_a2c_builder
    from isaacgymenvs.learning.networks import multihead_a2c_builder
    from isaacgymenvs.learning.networks import asymmetric_a2c_builder
    from isaacgymenvs.learning.networks import paco_a2c_builder
    from isaacgymenvs.learning.networks import grpo_builder
    from isaacgymenvs.learning.networks import td3_builder
    from isaacgymenvs.learning.networks import td3_simba_v2_builder
    from isaacgymenvs.learning.networks import simba_v2_a2c_builder
    from isaacgymenvs.learning.networks import bro_a2c_builder
    
    import isaacgymenvs

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env(**kwargs):
        # Create the original environment first
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        
        envs._freeze_rand_vec = True
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )

        return envs

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    task_observer = RLGPUAlgoObserver()
    if cfg.wandb_activate:
        wandb_observer = WandbAlgoObserver()
        task_observer = MultiObserver([task_observer, wandb_observer])

    # register new agent
    def build_agent(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('mt_a2c', lambda **kwargs : mt_a2c_agent.MTA2CAgent(**kwargs))
        runner.algo_factory.register_builder('ml_a2c', lambda **kwargs : ml_a2c_agent.MLA2CAgent(**kwargs))
        runner.algo_factory.register_builder('mt_sac', lambda **kwargs : mt_sac_agent.MTSACAgent(**kwargs))
        runner.algo_factory.register_builder('mt_sac_gradmani', lambda **kwargs : mt_sac_agent_gradmani.MTSACAgent(**kwargs))
        runner.algo_factory.register_builder('pqn', lambda **kwargs : pqn_agent.PQNAgent(**kwargs))
        runner.algo_factory.register_builder('grpo', lambda **kwargs : grpo_agent.MTGRPOAgent(**kwargs))
        runner.algo_factory.register_builder('td3', lambda **kwargs : td3_agent.TD3Agent(**kwargs))
        
        # register new agent for dynamic masking
        runner.algo_factory.register_builder('dynamic_mt_a2c', lambda **kwargs : DynamicMTA2CAgent(**kwargs))
        
        runner.player_factory.register_builder('mt_a2c', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        runner.player_factory.register_builder('ml_a2c', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        runner.player_factory.register_builder('mt_sac', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        runner.player_factory.register_builder('mt_sac_gradmani', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        runner.player_factory.register_builder('pqn', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        runner.player_factory.register_builder('grpo', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        runner.player_factory.register_builder('td3', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        
        # register new player for dynamic masking
        runner.player_factory.register_builder('dynamic_mt_a2c', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        
        return runner

    # register new networks
    def build_networks():
        model_builder.register('pq', lambda **kwargs : pq_builder.PQBuilder(**kwargs))
        model_builder.register('soft_modularized_pq', lambda **kwargs : soft_modularized_pq_builder.SoftModularizedPQBuilder(**kwargs))
        model_builder.register('soft_modularization_sac', lambda **kwargs : soft_modularization_sac_builder.SoftModularizationSACBuilder(**kwargs))
        model_builder.register('soft_modularization_a2c', lambda **kwargs : soft_modularization_a2c_builder.SoftModularizationA2CBuilder(**kwargs))
        model_builder.register('moore_sac', lambda **kwargs : moore_sac_builder.MooreSACBuilder(**kwargs))
        model_builder.register('moore_a2c', lambda **kwargs : moore_a2c_builder.MooreA2CBuilder(**kwargs))
        model_builder.register('care_a2c', lambda **kwargs : care_a2c_builder.CAREA2CBuilder(**kwargs))
        model_builder.register('multihead_a2c', lambda **kwargs : multihead_a2c_builder.MultiheadA2CBuilder(**kwargs))
        model_builder.register('asymmetric_a2c', lambda **kwargs : asymmetric_a2c_builder.AsymmetricA2CBuilder(**kwargs))
        model_builder.register('paco_a2c', lambda **kwargs : paco_a2c_builder.PACOA2CBuilder(**kwargs))
        model_builder.register('grpo', lambda **kwargs : grpo_builder.GRPOBuilder(**kwargs))
        model_builder.register('td3', lambda **kwargs : td3_builder.TD3Builder(**kwargs))
        model_builder.register('td3_simba_v2', lambda **kwargs : td3_simba_v2_builder.TD3SimbaV2Builder(**kwargs))
        model_builder.register('simba_v2_a2c', lambda **kwargs : simba_v2_a2c_builder.SimbaV2A2CBuilder(**kwargs))
        model_builder.register('bro_a2c', lambda **kwargs : bro_a2c_builder.BroA2CBuilder(**kwargs))
        model_builder.register('mt_models', lambda **kwargs : mt_models.MTModels(**kwargs))

    build_networks()

    # create runner and set the settings
    runner = build_agent(task_observer)
    runner.load(cfg_dict)
    runner.reset()

    # dump config dict
    experiment_dir = os.path.join('runs', cfg.train.params.config.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg_dict))

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma is not None else None
    })

if __name__ == "__main__":
    launch_rlg_hydra()