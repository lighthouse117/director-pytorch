import torch

from omegaconf import DictConfig
from networks.encoder import PixelEncoder
from networks.heads import PixelDecoderHead, RewardHead
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

        self.decoder = PixelDecoderHead(
            observation_shape=observation_shape,
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            config=config.decoder,
        ).to(device)

        self.reward_head = RewardHead(
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            config=config.reward_head,
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

        prior_distributions: list[torch.distributions.Distribution] = []
        posterior_distributions: list[torch.distributions.Distribution] = []
        deterministic_hs: list[torch.Tensor] = []

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
                    embeded_observation[:, t],
                    deterministic_h,
                )
            )

            # Append to list for calculating loss later
            prior_distributions.append(prior_stochastic_z_distribution)
            posterior_distributions.append(posterior_stochastic_z_distribution)
            deterministic_hs.append(deterministic_h)

            # Update previous states
            prev_deterministic_h = deterministic_h
            prev_stochastic_z = posterior_stochastic_z_distribution.rsample()

        # The state list has batch_size * chunk_length elements
        # We can regard them as a single batch of size (batch_size * chunk_length)
        # because we don't use recurrent model from here

        deterministic_hs = torch.cat(deterministic_hs, dim=0)
        # Get reparameterized samples of posterior z
        sampled_posteriors = torch.cat(
            [posterior.rsample() for posterior in posterior_distributions], dim=0
        )
        # Get gaussian distributions of reconstructed images
        reconstructed_obs_distributions: torch.distributions.Distribution = (
            self.decoder(
                deterministic_hs,
                sampled_posteriors,
            )
        )
        # Calculate reconstruction loss (log likelihood version)
        # How likely is the input image generated from the predicted distribution
        # reconstruction_loss = -reconstructed_obs_distributions.log_prob(
        #     # [batch_size, chunk_length, *observation_shape]
        #     # -> [batch_size * chunk_length, *observation_shape]
        #     transitions.observations.reshape(-1, *transitions.observations.shape[-3:])
        # ).mean()

        # Reconstruction loss (MSE version)
        reconstruction_loss = torch.nn.functional.mse_loss(
            reconstructed_obs_distributions.mean,
            transitions.observations.reshape(-1, *transitions.observations.shape[-3:]),
        )

        save_image(
            transitions.observations[0][0],
            "original.png",
        )
        save_image(
            reconstructed_obs_distributions.mean[0],
            "reconstructed.png",
        )

        # Convert list of distributions to a single distribution
        prior = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.cat([prior.mean for prior in prior_distributions], dim=0),
                torch.cat([prior.stddev for prior in prior_distributions], dim=0),
            ),
            reinterpreted_batch_ndims=1,
        )
        posterior = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.cat(
                    [posterior.mean for posterior in posterior_distributions], dim=0
                ),
                torch.cat(
                    [posterior.stddev for posterior in posterior_distributions], dim=0
                ),
            ),
            reinterpreted_batch_ndims=1,
        )

        # Calculate KL divergence loss
        # How different is the prior distribution from the posterior distribution
        kl_divergence_loss = torch.distributions.kl.kl_divergence(
            posterior, prior
        ).mean()

        # Calculate reward prediction loss
        reward_distribution: torch.distributions.Distribution = self.reward_head(
            deterministic_hs,
            sampled_posteriors,
        )
        reward_loss = -reward_distribution.log_prob(
            transitions.rewards.reshape(-1, *transitions.rewards.shape[-1:])
        ).mean()

        total_loss = reconstruction_loss + kl_divergence_loss + reward_loss

        # Update the parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
        self.optimizer.step()

        metrics = {
            "reconstruction_loss": round(reconstruction_loss.item(), 5),
            "kl_divergence_loss": round(kl_divergence_loss.item(), 5),
            "reward_loss": round(reward_loss.item(), 5),
            "total_loss": round(total_loss.item(), 5),
        }

        return metrics

    def evaluate(self):
        pass
