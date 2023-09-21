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

        self.encoder = PixelEncoder(
            observation_shape=observation_shape,
            embedded_observation_size=config.world_model.embedded_observation_size,
            config=config.encoder,
        ).to(device)

        # self.decoder = PixelDecoder(
        #     observation_shape=observation_shape,
        #     embedded_observation_size=config.world_model.embedded_observation_size,
        #     config=config.decoder,
        # ).to(device)

        self.optimiser = torch.optim.Adam(
            params=self.world_model.parameters(), lr=config.lr
        )

    def train(self):
        data_batch: TransitionBatch = self.replay_buffer.sample()
        self.world_model.train(data_batch)
