import torch

from omegaconf import DictConfig
from models.world_model import WorldModel
from utils.transition import Transition, TransitionBatch
from networks.pixel import PixelEncoder, PixelDecoder


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

        self.model_parameters = list(self.world_model.parameters())

        self.optimiser = torch.optim.Adam(
            params=self.model_parameters, lr=config.learning_rate
        )

    def train(self, transitions: TransitionBatch):
        # Update the world model
        self.world_model.train(transitions)

    def policy(self, observation: torch.Tensor) -> torch.Tensor:
        pass
