import gymnasium as gym
import numpy as np

from omegaconf import DictConfig
from agents.dreamer import DreamerAgent
from utils.transition import Transition
from utils.replay import ReplayBuffer
from utils.image import save_image, save_gif_video


class Driver:
    """
    A driver that runs N steps in an environment.
    """

    def __init__(
        self,
        env: gym.Env,
        agent: DreamerAgent,
        buffer: ReplayBuffer,
    ):
        self.env = env
        self.agent = agent
        self.buffer = buffer

    def run(self, max_steps: int = 1000, train_every: int = 5):
        """
        Run environment steps and train until reaching max_steps.
        """
        # Fill the replay buffer
        print("Filling the replay buffer...\n")
        obs, info = self.env.reset()
        while not self.buffer.is_full:
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            env_step += 1
            transition = Transition(
                observation=obs,
                action=action,
                next_observation=next_obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            self.buffer.add(transition)
            obs = next_obs
            if terminated or truncated:
                obs, info = self.env.reset()

        # Start training
        print(f"Start training for {max_steps} steps.")
        obs, info = self.env.reset()
        env_step = 0

        for total_step in range(max_steps):
            # action = self.agent.policy(obs)
            action = self.env.action_space.sample()

            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            env_step += 1

            transition = Transition(
                observation=obs,
                action=action,
                next_observation=next_obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

            # Add transition to the buffer
            self.buffer.add(transition)

            obs = next_obs

            if terminated or truncated:
                # print(f"Episode finished after {env_step} steps.")
                obs, info = self.env.reset()
                env_step = 0

            if total_step % train_every == 0:
                print(f"Training at step {total_step}.")
                # Get a batch from the buffer
                transitions = self.buffer.sample()

                # Train agent with the batch data
                self.agent.train(transitions)

        print("Finished.")
