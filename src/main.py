import torch
import hydra
import gymnasium as gym
import wandb

from omegaconf import DictConfig, OmegaConf
from agents.dreamer import DreamerAgent
from utils.replay import ReplayBuffer
from drivers.driver import Driver
from envs.wrappers import (
    ChannelFirstEnv,
    PixelEnv,
    ResizeImageEnv,
    ActionRepeatEnv,
    # BatchEnv,
)

# from envs.dmc import DMCPixelEnv
from envs.space import get_env_spaces


# Use hydra to load configs
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # Check device
    if config.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            config.device = "cpu"
        else:
            print("Using CUDA.")
    elif config.device == "mps":
        if not torch.backends.mps.is_available():
            print("Apple's MPS is not available. Using CPU instead.")
            config.device = "cpu"
        else:
            print("Using Apple's MPS.")
    elif config.device == "cpu":
        print("Using CPU.")
    print()

    print(" Config ".center(20, "="))
    print(OmegaConf.to_yaml(config))

    # ====================================================

    wandb.init(
        project="directorch",
        config=dict(config),
    )

    # Create environment
    env_name = config.environment.name
    env = ActionRepeatEnv(
        gym.make(env_name, render_mode="rgb_array"),
        repeat=config.environment.action_repeat,
    )
    # env = DMCPixelEnv(domain="cartpole", task="swingup")
    env = PixelEnv(env)
    env = ResizeImageEnv(
        env, (config.environment.image_width, config.environment.image_height)
    )
    env = ChannelFirstEnv(env)

    obs_shape, action_size, action_discrete = get_env_spaces(env)
    print(f"< {env_name} >")
    print(f"observation space: {obs_shape}")
    print(
        f"action space: {action_size} ({'discrete' if action_discrete else 'continuous'})"
    )

    # Create agent
    agent = DreamerAgent(
        observation_shape=obs_shape,
        action_size=action_size,
        action_discrete=action_discrete,
        device=config.device,
        config=config.agent,
    )

    # Create replay buffer
    buffer = ReplayBuffer(
        observation_shape=obs_shape,
        action_size=action_size,
        action_discrete=action_discrete,
        device=config.device,
        config=config.replay_buffer,
    )

    # The driver that runs the training loop
    driver = Driver(
        env=env,
        agent=agent,
        buffer=buffer,
    )

    # Start training
    driver.run(
        max_steps=config.total_steps,
        train_every=config.train_every,
    )


if __name__ == "__main__":
    main()
