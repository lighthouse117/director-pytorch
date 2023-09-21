import torch
import hydra
import pprint
import gymnasium as gym

from omegaconf import DictConfig, OmegaConf
from agents.dreamer import DreamerAgent
from utils.replay import ReplayBuffer
from drivers.driver import Driver
from gymnasium.wrappers import PixelObservationWrapper


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    print(" Config ".center(20, "="))
    print(OmegaConf.to_yaml(config))

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

    # ----------------------------------------

    # Create environment
    env_name = config.environment.name
    env = PixelObservationWrapper(gym.make(env_name, render_mode="rgb_array"))
    obs_shape = env.observation_space.shape
    action_size = env.action_space.n
    print(f"<{env_name}>")
    print(f"observation space: {obs_shape}")
    print(f"action size: {action_size}")

    # Create agent
    agent = DreamerAgent(
        observation_shape=obs_shape,
        action_size=action_size,
        device=config.device,
        config=config.agent,
    )

    # Create replay buffer
    buffer = ReplayBuffer(
        observation_shape=obs_shape,
        action_size=action_size,
        device=config.device,
        config=config.replay_buffer,
    )

    driver = Driver(env=env, agent=agent, buffer=buffer)

    driver.run(max_steps=1000)


if __name__ == "__main__":
    main()
