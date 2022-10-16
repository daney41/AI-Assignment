import argparse
import os

import gym
import math
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from dqn_common import DqnNet, epsilon_by_frame
from lib.experience_buffer import ExperienceBuffer, Experience
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", default="CartPole-v1", help="Full name of the environment, e.g. CartPole-v1, LunarLander-v2, etc.")
parser.add_argument("-c", "--config_file", default="config/dqn.yaml", help="Config file with hyper-parameters")
args = parser.parse_args()

# Hyperparameters for the requried environment
hypers = yaml.load(open(args.config_file), Loader=yaml.FullLoader)

if args.env not in hypers:
    raise Exception(f'Hyper-parameters not found for env {args.env} - please add it to the config file (config/q-learning.yaml)')
params = hypers[args.env]

env = gym.make(args.env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = DqnNet(obs_size=env.observation_space.shape[0],
             hidden_size=params['hidden_size'], hidden_size2=params['hidden_size2'],
             n_actions=env.action_space.n).to(device)
target_net = DqnNet(obs_size=env.observation_space.shape[0],
                    hidden_size=params['hidden_size'], hidden_size2=params['hidden_size2'],
                    n_actions=env.action_space.n).to(device)
print(net)

# writer = SummaryWriter(comment="-CartPoleScratch")

buffer = ExperienceBuffer(int(params['replay_size']), device)

optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

frame_idx = 0
max_reward = -math.inf
all_rewards = []
losses = []
episode_reward = 0
r100 = -math.inf
episode_start = time.time()
start = time.time()
episode_frame = 0
episode_no = 0

state, _ = env.reset()

def calculate_loss(net, target_net):
  states_v, actions_v, rewards_v, dones_v, next_states_v = buffer.sample(params['batch_size'])

  # get the Q value of the state - i.e. Q value for each action possible in that state
  # in CartPole there are 2 actions so this will be tensor of (2, BatchSize)
  Q_s = net.forward(states_v)

  # now we need the state_action_values for the actions that were selected (i.e. the action from the tuple)
  # actions tensor is already {100, 1}, i.e. unsqeezed so we don't need to unsqueeze it again
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
  expected_state_action_values = rewards_v + next_state_values.detach() * params['gamma'] * (1 - dones_v)

  loss = F.mse_loss(state_action_values, expected_state_action_values)

  optimizer.zero_grad()
  loss.backward()
  # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
  optimizer.step()

  return loss


while True:
  frame_idx += 1
  episode_no += 1

  # calculate the value of decaying epsilon
  epsilon = epsilon_by_frame(frame_idx, params)
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
  new_state, reward, terminated, truncated, _ = env.step(action)
  is_done = terminated or truncated
  episode_reward += reward

  # store the transition in the experience replay buffer
  exp = Experience(state, action, reward, is_done, new_state)
  buffer.append(exp)
  state = new_state

  # when the episode is done, reset and update progress
  if is_done:
    done_reward = episode_reward
    all_rewards.append(episode_reward)
    state, _ = env.reset()
    if episode_reward > max_reward:
      max_reward = episode_reward

    if len(all_rewards) > 10 and len(losses) > 10:
      r100 = np.mean(all_rewards[-100:])
      l100 = np.mean(losses[-100:])
      fps = (frame_idx - episode_frame) / (time.time() - episode_start)
      print(f"Frame: {frame_idx}: Episode: {episode_no}, R100: {r100}, MaxR: {max_reward}, R: {episode_reward}, FPS: {fps}, L100: {l100}, Epsilon: {epsilon}")

    episode_reward = 0
    episode_frame = frame_idx
    episode_start = time.time()

  if len(buffer) < params['replay_size_start']:
    continue

  # do the learning
  loss = calculate_loss(net, target_net)
  losses.append(loss.item())

  if frame_idx % params['target_net_sync'] == 0:
    target_net.load_state_dict(net.state_dict())

  if r100 > params['stopping_reward']:
    print("Finished training")

    name = f"{args.env}_DQN_act_net_%+.3f_%d.dat" % (r100, frame_idx)
    torch.save(net.state_dict(), os.path.join(params['save_path'], name))

    break

  if frame_idx > params['max_frames']:
    print(f"Ran out of time at {time.time() - start}")
    break
