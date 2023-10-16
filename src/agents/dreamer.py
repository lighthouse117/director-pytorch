import torch

from omegaconf import DictConfig
from models.world_model import WorldModel
from utils.transition import Transition, TransitionSequenceBatch


class DreamerAgent:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        device: str,
        config: DictConfig,
    ):
        self.config = config
        self.device = device

        self.world_model = WorldModel(
            observation_shape=observation_shape,
            action_size=action_size,
            device=device,
            config=config.world_model,
        )

    def train(self, transitions: TransitionSequenceBatch) -> dict:
        # Update the world model
        posterior_zs, deterministic_hs, metrics = self.world_model.train(transitions)

        # Update the agent

        return metrics

    def policy(self, observation: torch.Tensor) -> torch.Tensor:
        pass
