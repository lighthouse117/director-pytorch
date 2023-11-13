import torch

from torch import nn
from omegaconf import DictConfig
from networks.actor_critic import Actor, Critic


class Policy(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        action_discrete: bool,
        embedded_observation_size: int,
        deterministic_state_size: int,
        stochastic_state_size: int,
        device: str,
        config: DictConfig,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.action_discrete = action_discrete
        self.embedded_observation_size = embedded_observation_size
        self.deterministic_state_size = deterministic_state_size
        self.stochastic_state_size = stochastic_state_size
        self.device = device
        self.config = config

        self.actor = Actor(
            deterministic_state_size=deterministic_state_size,
            stochastic_state_size=stochastic_state_size,
            action_size=action_size,
            config=config.actor,
        ).to(device)

        self.critic = Critic(
            deterministic_state_size=deterministic_state_size,
            stochastic_state_size=stochastic_state_size,
            config=config.critic,
        ).to(device)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)

    def loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        # Compute the loss for the actor
        actor_loss = self.actor.loss(
            observation, action, reward, next_observation, done
        )

        # Compute the loss for the critic
        critic_loss = self.critic.loss(
            observation, action, reward, next_observation, done
        )

        return actor_loss + critic_loss
