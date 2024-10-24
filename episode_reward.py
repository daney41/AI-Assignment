import pickle
import matplotlib.pyplot as plt

# read the data from .pkl file
with open('Q7-sync-1e2-LunarLander-v3.pkl', 'rb') as f1, open('Q7-sync-1e3-LunarLander-v3.pkl', 'rb') as f2, open('Q7-sync-1e4-LunarLander-v3.pkl', 'rb') as f3, open('Q4-lr-0.01-CartPole-v0.pkl', 'rb') as f4:
    data1 = pickle.load(f1)
    data2 = pickle.load(f2)
    data3 = pickle.load(f3)
    data4 = pickle.load(f4)

# retrieve saved data
all_rewards_1 = data1['all_rewards']
r100_plot_1 = data1['r100_plot']
all_rewards_2 = data2['all_rewards']
r100_plot_2 = data2['r100_plot']
all_rewards_3 = data3['all_rewards']
r100_plot_3 = data3['r100_plot']
# all_rewards_4 = data4['all_rewards']
# r100_plot_4 = data4['r100_plot']

plt.figure(figsize=(15, 5))

# plt.plot(all_rewards_1, label='Rewards LR=0.001', color='gray')
plt.plot(r100_plot_1, label='R100 sync=0.001', color='red')
# plt.plot(all_rewards_2, label='Rewards LR=0.0001', color='darkgray')
plt.plot(r100_plot_2, label='R100 sync=0.0001', color='dodgerblue')
# plt.plot(all_rewards_3, label='Rewards LR=0.00001')
plt.plot(r100_plot_3, label='R100 sync=0.00001', color='orange')
# plt.plot(all_rewards_4, label='Rewards LR=0.01')
# plt.plot(r100_plot_4, label='R100 LR=0.01')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Comparison of Learning Rates')
plt.legend()
plt.show()

# plot the data
# def plot_results(all_rewards, r100_plot):
#     plt.figure(figsize=(10, 9))
#
#     plt.suptitle('CartPole-v0-sync', fontsize=16)
#
#     # Rewards
#     plt.plot(all_rewards, label='Rewards')
#
#     # R100
#     plt.plot(range(len(r100_plot)), r100_plot, label='R100', color='orange')
#
#     plt.title('Rewards and R100')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
# # Call the plotting function
# plot_results(all_rewards, r100_plot)