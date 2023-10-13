import torch
import numpy as np
from torch import nn

from omegaconf import DictConfig


class PixelEncoder(nn.Module):
    """
    Encodes an image observation into a embedded observation (o) using CNNs.

    o ~ p(o|x)

    ### Input:
    - Image observation (x)

    ### Output:
    - Embedded observation (o)
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        embedded_observation_size: int,
        config: DictConfig,
    ):
        super().__init__()

        self.observation_shape = observation_shape
        self.embedded_observation_size = embedded_observation_size
        self.config = config

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
            print(
                f"Reshaping CNN output from {self.cnn_output_size} to {embedded_observation_size}\n"
            )
            self.fc = nn.Linear(
                in_features=self.cnn_output_size,
                out_features=embedded_observation_size,
            )
        else:
            self.fc = nn.Identity()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.convs(observation)
        # Flatten the values after the first batch dimension
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
    """
    Decodes latent states (deterministic and stochastic) into an image observation using CNNs.

    x ~ p(x|h, z)

    ### Input:
    - Deterministic state (h)
    - Stochastic state (z)

    ### Output:
    - Image observation (x)
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        deterministic_state_size: int,
        stochastic_state_size: int,
        config: DictConfig,
    ):
        super().__init__()

        self.observation_shape = observation_shape
        self.deteministic_state_size = deterministic_state_size
        self.stochastic_state_size = stochastic_state_size
        self.config = config

        self.convs = nn.Sequential(
            nn.Linear(
                in_features=deterministic_state_size + stochastic_state_size,
                out_features=config.depth * 8 * 2 * 2,
            ),
        )

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.fc(latent_state)
        x = x.reshape(-1, self.depth * 8, 2, 2)
        x = self.convs(x)
        return x
