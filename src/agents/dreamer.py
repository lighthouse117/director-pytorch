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
            embedded_observation_size=config.embedded_observation_size,
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            device=device,
            config=config.world_model,
        )

        self.policy = Policy(
            observation_shape=observation_shape,
            action_size=action_size,
            action_discrete=action_discrete,
            embedded_observation_size=config.embedded_observation_size,
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            device=device,
            config=config.policy,
        )

    def train(self, transitions: TransitionSequenceBatch) -> dict:
        metrics = {}

        # Update the world model
        stochastic_zs, deterministic_hs, met = self.world_model.train(transitions)
        # print("stochastic_posterior_zs.shape", stochastic_zs.shape)
        # print("deterministic_hs.shape", deterministic_hs.shape)

        metrics.update(met)

        # Imagine next states and rewards using the current policy
        (
            imagined_stoch_zs,
            imagined_deter_hs,
            imagined_rewards,
        ) = self.world_model.imagine(
            stochastic_zs=stochastic_zs,
            deterministic_hs=deterministic_hs,
            horizon=self.config.imagination_horizon,
            actor=self.policy.actor,
        )

        # Update the policy
        met = self.policy.train(
            stochastic_zs=imagined_stoch_zs,
            deterministic_hs=imagined_deter_hs,
            rewards=imagined_rewards,
        )

        metrics.update(met)

        return metrics

    def select_aciton(self, observation: torch.Tensor) -> torch.Tensor:
        if self.action_discrete:
            action = random.randint(0, self.action_size - 1)
        else:
            action = np.random.uniform(-1, 1, self.action_size)

        return action
