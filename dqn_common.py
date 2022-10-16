import math

import torch.nn as nn

# Define the network structure - in this case 2 hidden layers (CartPole can be solved faster with a single hidden layer)
class DqnNet(nn.Module):
  def __init__(self, obs_size, hidden_size, hidden_size2, n_actions):
    super(DqnNet, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(obs_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size2),
      nn.ReLU(),
      nn.Linear(hidden_size2, n_actions)
    )

  def forward(self, x):
    return self.net(x.float())


def epsilon_by_frame(frame_idx, params):
  return params['epsilon_final'] + (params['epsilon_start'] - params['epsilon_final']) * math.exp(-1.0 * frame_idx / params['epsilon_decay'])


