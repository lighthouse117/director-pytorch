import torch
import hydra
import pprint
import gymnasium as gym

from omegaconf import DictConfig, OmegaConf
from utils.replay import ReplayBuffer
from utils.transition import Transition
from networks.pixel import PixelEncoder


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

    env = gym.make("CartPole-v1")

    obs_shape = env.observation_space.shape
    action_size = env.action_space.n

    obs, info = env.reset()

    buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        batch_size=config.batch_size,
        observation_shape=obs_shape,
        action_size=action_size,
    )

    while buffer.full is False:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        transition = Transition(
            observation=obs,
            action=action,
            next_observation=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        buffer.add(transition)

        obs = next_obs

        if terminated or truncated:
            obs, info = env.reset()

    x = torch.randn(16, 3, 64, 64)

    encoder = PixelEncoder(
        observation_shape=x.shape[1:],
    )

    print(encoder(x).shape)


if __name__ == "__main__":
    main()
