import argparse

import gym
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.experience_buffer import ExperienceBuffer, Experience
import os

# Define the network structure - in this case 2 hidden layers (CartPole can be solved faster with a single hidden layer)
class Net(nn.Module):
  def __init__(self, obs_size, hidden_size, hidden_size2, n_actions):
    super(Net, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(obs_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size2),
      nn.ReLU(),
      nn.Linear(hidden_size2, n_actions)
    )

  def forward(self, x):
    return self.net(x.float())

def epsilon_by_frame(frame_idx):
  return EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * math.exp(-1.0 * frame_idx / EPSILON_DECAY)

device = torch.device("cpu")

GAMMA=0.9
BATCH_SIZE = 64
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000

EPSILON_DECAY = 5000
EPSILON_FINAL = 0.02
EPSILON_START = 1.0

LEARNING_RATE = 1e-3

TARGET_NET_SYNC = 10 ** 3

STOP_REWARD = 195
# this seems to have no effect - it still spins a process for every core and maxes it out to 99% utiliziation
os.environ['OMP_NUM_THREADS'] = "1"
env = gym.make("CartPole-v1")
# env.render()

net = Net(obs_size=env.observation_space.shape[0], hidden_size=128, hidden_size2=128, n_actions=env.action_space.n).to(device)
target_net = Net(obs_size=env.observation_space.shape[0], hidden_size=128, hidden_size2=128, n_actions=env.action_space.n).to(device)
print(net)

# writer = SummaryWriter(comment="-CartPoleScratch")

buffer = ExperienceBuffer(REPLAY_SIZE, device)

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

frame_idx = 0
max_reward = -10000
all_rewards = []
losses = []
episode_reward = 0
r100 = 0
episode_start = time.time()
start = time.time()
episode_frame = 0

state = env.reset()

def calculate_loss(net, target_net):
  states_v, actions_v, rewards_v, dones_v, next_states_v = buffer.sample(BATCH_SIZE)

  # get the Q value of the state - i.e. Q value for each action possible in that state
  # in CartPole there are 2 actions so this will be tensor of (2, BatchSize)
  Q_s = net.forward(states_v)

  # now we need the state_action_values for the actions that were selected (i.e. the action from the tuple)
  # unlike my normal DQN, the actions tensor is already {100, 1}, i.e. unsqeezed so we don't need to unsqueeze it again
  # because the Q_s has one row per sample and the actions will be use as indices to choose the value from each row
  # lastly, because the gather will return a column and we need a row, we will squeeze it
  # gather on dim 1 means on rows
  state_action_values = Q_s.gather(1, actions_v.type(torch.int64).unsqueeze(-1)).squeeze(-1)

  # now we need Q_s_prime_a - i.e. the next state values
  # we get them from the target net
  # because there are 2 actions, we get a tensor of (2, BatchSize)
  # and because it's Sarsa max, we are taking the max
  # .max(1) will find maximum for each row and return a tuple (values, indices) - we need values so get<0>
  next_state_values = target_net.forward(next_states_v).max(1)[0]

  # calculate expected action values - discounted value of the state + reward
  expected_state_action_values = rewards_v + next_state_values.detach() * GAMMA * (1 - dones_v)

  loss = F.mse_loss(state_action_values, expected_state_action_values)

  optimizer.zero_grad()
  loss.backward()
  torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
  optimizer.step()

  return loss


while True:
  frame_idx += 1

  # calculate the value of decaying epsilon
  epsilon = epsilon_by_frame(frame_idx)
  if np.random.random() < epsilon:
    # explore
    action = env.action_space.sample()
  else:
    # exploit
    state_a = np.array([state], copy=False)
    state_v = torch.tensor(state_a).to(device)
    q_vals_v = net(state_v)
    _, act_v = torch.max(q_vals_v, dim=1)
    action = int(act_v.item())
    # print(action)

  # take step in the environment
  new_state, reward, is_done, _ = env.step(action)
  episode_reward += reward

  # store the transition in the experience replay buffer
  exp = Experience(state, action, reward, is_done, new_state)
  buffer.append(exp)
  state = new_state

  # when the episode is done, reset and update progress
  if is_done:
    done_reward = episode_reward
    all_rewards.append(episode_reward)
    state = env.reset()
    if episode_reward > max_reward:
      max_reward = episode_reward

    if len(all_rewards) > 101 and len(losses) > 101:
      r100 = np.mean(all_rewards[-100:])
      l100 = np.mean(losses[-100:])
      fps = (frame_idx - episode_frame) / (time.time() - episode_start)
      print(f"Frame: {frame_idx}: R100: {r100}, MaxR: {max_reward}, R: {episode_reward}, FPS: {fps}, Epsilon: {epsilon}, L100: {l100}")

    episode_reward = 0
    episode_frame = frame_idx
    episode_start = time.time()

  if len(buffer) < REPLAY_START_SIZE:
    continue

  # do the learning
  loss = calculate_loss(net, target_net)
  losses.append(loss.item())

  if frame_idx % TARGET_NET_SYNC == 0:
    target_net.load_state_dict(net.state_dict())

  if r100 > 195:
    print("Finished training")
    break

  if frame_idx > 100000:
    print(f"Ran out of time at {time.time() - start}")
    break
