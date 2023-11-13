import torch
from torch import nn
from torch import Tensor

from omegaconf import DictConfig


class Actor(nn.Module):
    """
    Actor network for SAC.
    """

    def __init__(
        self,
        deterministic_state_size: int,
        stochastic_state_size: int,
        action_size: int,
        config: DictConfig,
    ):
        super().__init__()

        self.config = config

        self.network = nn.Sequential(
            nn.Linear(
                deterministic_state_size + stochastic_state_size,
                config.hidden_size,
            ),
            nn.ELU(),
            nn.Linear(
                config.hidden_size,
                config.hidden_size,
            ),
            nn.ELU(),
            nn.Linear(
                config.hidden_size,
                action_size,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        deter_h: Tensor,
        stoch_z: Tensor,
    ) -> Tensor:
        x = torch.cat([deter_h, stoch_z], dim=-1)
        x = self.network(x)

        onehot_distribution = torch.distributions.OneHotCategoricalStraightThrough(
            logits=x
        )

        # Use straight-through trick for discrete actions
        # sample = sample + prob - prob.detach()
        # (Gumbel-Softmax also works but needs hyperparameter tuning)
        action = onehot_distribution.rsample()

        return action


class Critic(nn.Module):
    """
    Critic network for SAC.
    """

    def __init__(
        self,
        deterministic_state_size: int,
        stochastic_state_size: int,
        config: DictConfig,
    ):
        super().__init__()

        self.config = config

        self.network = nn.Sequential(
            nn.Linear(
                deterministic_state_size + stochastic_state_size,
                config.hidden_size,
            ),
            nn.ELU(),
            nn.Linear(
                config.hidden_size,
                config.hidden_size,
            ),
            nn.ELU(),
            nn.Linear(
                config.hidden_size,
                1,
            ),
        )

    def forward(
        self,
        deter_h: Tensor,
        stoch_z: Tensor,
        action: Tensor,
    ) -> Tensor:
        x = torch.cat([deter_h, stoch_z, action], dim=-1)
        x = self.network(x)
        return x
