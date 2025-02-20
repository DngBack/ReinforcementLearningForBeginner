import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    A neural network for policy-based reinforcement learning.

    This network takes an input state and outputs a probability distribution over actions.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initializes the Policy Network.

        Args:
            input_dim (int): The dimensionality of the input state.
            output_dim (int): The number of possible actions (output size).
        """
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),  # Increase layer size
            nn.ReLU(),
            nn.Linear(256, 128),  # Add an extra hidden layer
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: Probability distribution over actions.
        """
        return self.fc(state)
