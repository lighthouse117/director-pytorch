import torch

from omegaconf import DictConfig
from networks.pixel import PixelEncoder, PixelDecoder
from networks.rssm import RepresentationModel, TransitionModel, RecurrentModel
from utils.transition import TransitionSequenceBatch
from torchvision.utils import save_image


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

        # Models
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

        # Optimizer
        self.optimizer = torch.optim.Adam(
            params=self.parameters(), lr=config.learning_rate
        )

    def train(self, transitions: TransitionSequenceBatch) -> dict:
        """
        Update the world model using transition data.
        """
        # Encode observations
        embeded_observation = self.encoder(transitions.observations)

        # Initial input for recurrent model
        prev_deterministic_h = torch.zeros(
            len(transitions), self.config.deterministic_state_size
        ).to(self.device)
        prev_stochastic_z = torch.zeros(
            len(transitions), self.config.stochastic_state_size
        ).to(self.device)

        chunk_length = len(transitions.observations[0])

        # Iterate over timesteps of a chunk
        for t in range(chunk_length):
            # Predict deterministic state h_t from h_t-1, z_t-1, and a_t-1
            deterministic_h = self.recurrent_model(
                prev_stochastic_z,
                transitions.actions[:, t],
                prev_deterministic_h,
            )

            # Predict stochastic state z_t (gaussian) from h_t without o_t
            # (called Prior because it is before seeing observation)
            prior_stochastic_z_distribution = self.transition_model(deterministic_h)

            # Predict stochastic state z_t using both h_t and o_t
            # (called Posterior because it is after seeing observation)
            posterior_stochastic_z_distribution: torch.distributions.Distribution = (
                self.representation_model(
                    embeded_observation,
                    deterministic_h,
                )
            )

            # Gaussian distribution of reconstructed image
            reconstructed_obs_distribution: torch.distributions.Distribution = self.decoder(
                deterministic_h,
                # Get reparameterized sample of z
                posterior_stochastic_z_distribution.rsample(),
            )

        save_image(
            transitions.observations,
            "original.png",
        )
        save_image(
            reconstructed_obs_distribution.rsample(),
            "reconstructed.png",
        )

        # Calculate reconstruction loss
        # How likely is the input image generated from the predicted distribution
        reconstruction_loss = -reconstructed_obs_distribution.log_prob(
            transitions.observations
        ).mean()

        # Calculate KL divergence loss
        # How different is the prior distribution from the posterior distribution
        kl_divergence_loss = torch.distributions.kl.kl_divergence(
            posterior_stochastic_z_distribution, prior_stochastic_z_distribution
        ).mean()

        total_loss = reconstruction_loss + kl_divergence_loss

        # Update the parameters
        self.optimizer.zero_grad()
        reconstruction_loss.backward()
        self.optimizer.step()

        metrics = {
            "reconstruction_loss": reconstruction_loss.item(),
            # "kl_divergence_loss": kl_divergence_loss.item(),
            # "total_loss": total_loss.item(),
        }

        return metrics

    def evaluate(self):
        pass
