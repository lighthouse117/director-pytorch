import torch
import random
import numpy as np

from omegaconf import DictConfig
from models.world_model import WorldModel
from utils.transition import Transition, TransitionSequenceBatch
from models.policy import Policy


class DreamerAgent:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        action_discrete: bool,
        device: str,
        config: DictConfig,
    ):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.action_discrete = action_discrete
        self.config = config
        self.device = device

        self.world_model = WorldModel(
            observation_shape=observation_shape,
            action_size=action_size,
            action_discrete=action_discrete,
            embeded_observation_size=config.embeded_observation_size,
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            device=device,
            config=config.world_model,
        )

        self.critic = Policy(
            observation_shape=observation_shape,
            action_size=action_size,
            action_discrete=action_discrete,
            embeded_observation_size=config.embeded_observation_size,
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            device=device,
            config=config.policy,
        )

    def train(self, transitions: TransitionSequenceBatch) -> dict:
        # Update the world model
        posterior_zs, deterministic_hs, metrics = self.world_model.train(transitions)

        # Predict

        # Update the agent
        self.critic

        return metrics

    def policy(self, observation: torch.Tensor) -> torch.Tensor:
        if self.action_discrete:
            action = random.randint(0, self.action_size - 1)
        else:
            action = np.random.uniform(-1, 1, self.action_size)

        return action
