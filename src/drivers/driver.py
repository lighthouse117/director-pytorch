import gymnasium as gym

from agents.dreamer import DreamerAgent
from utils.transition import Transition
from utils.replay import ReplayBuffer


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

    def run(self, max_steps: int = 1000):
        """
        Run environment steps until reaching max_steps.
        """
        obs, info = self.env.reset()

        for step in range(max_steps):
            # action = self.agent.policy(obs)
            action = self.env.action_space.sample()

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            transition = Transition(
                observation=obs,
                action=action,
                next_observation=next_obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

            # self.agent.train(transition)

            obs = next_obs

            if terminated or truncated:
                break
