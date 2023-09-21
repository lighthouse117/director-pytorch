import torch
import numpy as np
from torch import nn

from omegaconf import DictConfig


class PixelEncoder(nn.Module):
    """
    Encodes an image observation into a embedded observation (o) using CNNs.
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        embedded_observation_size: int,
        config: DictConfig,
    ):
        super().__init__()

        self.observation_shape = observation_shape

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=observation_shape[0],
                out_channels=config.depth,
                kernel_size=config.kernel_size,
                stride=config.stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.depth,
                out_channels=config.depth * 2,
                kernel_size=config.kernel_size,
                stride=config.stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.depth * 2,
                out_channels=config.depth * 4,
                kernel_size=config.kernel_size,
                stride=config.stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.depth * 4,
                out_channels=config.depth * 8,
                kernel_size=config.kernel_size,
                stride=config.stride,
            ),
            nn.ReLU(),
        )

        if self.cnn_output_size != embedded_observation_size:
            self.fc = nn.Linear(self.cnn_output_size, embedded_observation_size)
        else:
            self.fc = nn.Identity()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.convs(observation)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

    @property
    def cnn_output_size(self) -> int:
        dummy_input = torch.empty(1, *self.observation_shape)
        with torch.no_grad():
            dummy_input = self.convs(dummy_input)
        output_size = np.prod(dummy_input.shape[1:])
        return output_size


class PixelDecoder(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        latent_size=128,
        depth=32,
        kernel_size=4,
        stride=2,
    ):
        super().__init__()

        self.observation_shape = observation_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride

        self.fc = nn.Sequential(
            nn.Linear(latent_size, self.cnn_input_size),
            nn.ReLU(),
        )

        self.convs = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=depth * 8,
                out_channels=depth * 4,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=depth * 4,
                out_channels=depth * 2,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=depth * 2,
                out_channels=depth,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=depth,
                out_channels=observation_shape[0],
                kernel_size=kernel_size,
                stride=stride,
            ),
            # nn.Sigmoid(),
        )

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.fc(latent_state)
        x = x.reshape(-1, self.depth * 8, 2, 2)
        x = self.convs(x)
        return x
