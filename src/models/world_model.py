import torch

from omegaconf import DictConfig
from networks.rssm import RepresentationModel, TransitionModel, RecurrentModel


class WorldModel(torch.nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        device: str,
        config: DictConfig,
    ):
        super().__init__()

        self.device = device
        self.config = config

        self.representation_model = RepresentationModel(
            embeded_observation_size=config.embedded_observation_size,
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            config=config.representation_model,
        ).to(device)

        self.transition_model = TransitionModel(
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            config=config.transition_model,
        ).to(device)

        self.recurrent_model = RecurrentModel(
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            action_size=action_size,
            config=config.recurrent_model,
        ).to(device)

    def train(self):
        pass

    def evaluate(self):
        pass
