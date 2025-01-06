import torch
import torch.nn as nn

class ResNet(nn.Module):
  """
  Input: game state in shape (3, rows, columns). One dimension for player 1 pieces,
         one for player 2 pieces, and one for empty spaces.
  Output: policy in shape (action_size) and value in shape (1)
  """

  def __init__(self, game, num_resBlocks, num_hidden, device):
    super().__init__()

    self.device = device
    self.startBlock = nn.Sequential(
        nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_hidden),
        nn.ReLU()
    )

    # Stack of residual blocks
    self.backBone = nn.ModuleList(
        [ResBlock(num_hidden) for i in range(num_resBlocks)]
    )

    # Policy head outputs action probabilities
    self.policyHead = nn.Sequential(
        nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * game.rows * game.columns, game.action_size)
    )

    # Value head estimates game state value between [-1,1]
    self.valueHead = nn.Sequential(
        nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3 * game.rows * game.columns, 1),
        nn.Tanh()
    )

    self.to(device)

  def forward(self, x):
    x = self.startBlock(x)
    for resBlock in self.backBone:
      x = resBlock(x)
    policy = self.policyHead(x)
    value = self.valueHead(x)

    return policy, value


class ResBlock(nn.Module):
  """
  Residual block with two convolutional layers and skip connection
  """

  def __init__(self, num_hidden):
     super().__init__()
     self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
     self.bn1 = nn.BatchNorm2d(num_hidden)
     self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
     self.bn2 = nn.BatchNorm2d(num_hidden)
     self.relu = nn.ReLU()

  def forward(self, x):
    residual = x
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.bn2(self.conv2(x))
    x += residual
    x = self.relu(x)

    return x
