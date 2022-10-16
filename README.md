# Basic DQN

This repo has some basic DQN examples.

### Requirements
I use conda to manage virtual environments so you will need [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or just install all the packages manually).
To install the dependencies with conda use:

    conda env create -f environment.yml

This way, PyTorch without GPU will be installed. If you have a GPU and want a GPU version, follow [these instructions](https://pytorch.org/get-started/locally/).

### Running

    python dqn_cartpole.py

With the default hyper-params it should start learning at about 13k frames and it should reach R100 of 195 at about 40k.

The hyper-parameters are hard-coded to make it easier to follow, but they should be moved to a config file.

#### [OpenAI Gym](https://www.gymlibrary.dev/) Environments

[dqn_gym.py](dqn_gym.py) has the same code but the hyper-params are moved to the [config file](config/dqn.yaml) for easier experimentation.
