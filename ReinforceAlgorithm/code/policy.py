import torch.nn as nn
from torch import Tensor


class PolicyNetwork(nn.Module):
    """
    A neural network policy used in the REINFORCE algorithm,
    mapping input states to a probability distribution over actions.

    Attributes:
        fc (nn.Sequential): A fully connected network with two linear layers,
                            ReLU activation, and Softmax at the output to produce
                            a probability distribution.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initializes the policy network.

        Args:
            input_dim (int): The size of the input (number of state dimensions).
            output_dim (int): The number of possible actions.
        """
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1),  # Probability distribution over actions
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (Tensor): The input state tensor with shape (batch_size, input_dim).

        Returns:
            Tensor: A probability distribution over actions with shape (batch_size, output_dim).
        """
        return self.fc(x)
