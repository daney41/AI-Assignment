import pickle
import matplotlib.pyplot as plt

# read the data from .pkl file
with open('Q2-CartPole-v1.pkl', 'rb') as f:
    data = pickle.load(f)

# retrieve saved data
all_rewards = data['all_rewards']
r100_plot = data['r100_plot']

# plot the data
def plot_results(all_rewards, r100_plot):
    plt.figure(figsize=(10, 9))

    plt.suptitle('CartPole-v1', fontsize=16)

    # Rewards
    plt.plot(all_rewards, label='Rewards')

    # R100
    plt.plot(range(len(r100_plot)), r100_plot, label='R100', color='orange')

    plt.title('Rewards and R100')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_results(all_rewards, r100_plot)