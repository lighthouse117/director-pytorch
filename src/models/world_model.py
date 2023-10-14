import torch

from omegaconf import DictConfig
from networks.pixel import PixelEncoder, PixelDecoder
from networks.rssm import RepresentationModel, TransitionModel, RecurrentModel
from utils.transition import TransitionBatch


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

        self.encoder = PixelEncoder(
            observation_shape=observation_shape,
            embedded_observation_size=config.embedded_observation_size,
            config=config.encoder,
        ).to(device)

        self.decoder = PixelDecoder(
            observation_shape=observation_shape,
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            config=config.decoder,
        ).to(device)

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

    def train(self, transitions: TransitionBatch):
        """
        Update the world model using transition data.
        """
        # Encode observations
        embeded_observation = self.encoder(transitions.observations)

        prev_deterministic_h = torch.zeros(
            len(transitions), self.config.deterministic_state_size
        ).to(self.device)
        prev_stochastic_z = torch.zeros(
            len(transitions), self.config.stochastic_state_size
        ).to(self.device)

        # Predict deterministic state h_t from h_t-1, z_t-1, and a_t-1
        deterministic_h = self.recurrent_model(
            prev_stochastic_z, transitions.actions, prev_deterministic_h
        )

        # Predict stochastic state z_t from h_t without o_t
        # (called Prior because it is before seeing observation)
        prior_stochastic_z = self.transition_model(deterministic_h)

        # Predict stochastic state z_t using both h_t and o_t
        # (called Posterior because it is after seeing observation)
        posterior_stochastic_z = self.representation_model(
            embeded_observation, deterministic_h
        )

    def evaluate(self):
        pass
