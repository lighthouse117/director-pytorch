import torch
from torch import nn
from torch import Tensor


class RSSM(nn.Module):
    """Recurrent State Space Model."""

    def __init__(
        self,
    ):
        super().__init__()
        self.representation_model = RepresentationModel()
        self.transition_model = TransitionModel()
        self.recurrence_model = RecurrenceModel()


class RepresentationModel(nn.Module):
    """
    Infer the stochastic state (z) from the embed observation (o) and the deterministic state (h).

    z ~ p(z|o, h)

    - Embed observation (o) : observation after encoding
    - Deterministic state (h)
    - Stochastic state (z)
    """

    def __init__(
        self,
        embeded_observation_size: int,
        deterministic_state_size: int,
        stochastic_state_size: int,
        hidden_size: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embeded_observation_size + deterministic_state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, stochastic_state_size * 2),
            # Outputs mean and std for gaussian distribution
        )

    def forward(self, embeded_o: Tensor, deter_h: Tensor) -> Tensor:
        x = torch.cat([embeded_o, deter_h], dim=-1)
        x = self.network(x)

        # Split concatenated output into mean and standard deviation
        mean, std = torch.chunk(x, 2, dim=-1)

        # Apply softplus to std to ensure it is positive
        # Add min_std for numerical stability
        min_std = 0.1
        std = torch.nn.functional.softplus(std) + min_std

        # Create gaussian distribution
        distribution = torch.distributions.Normal(mean, std)
        return distribution