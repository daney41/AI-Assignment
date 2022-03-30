# Basic DQN

This repo has some basic DQN examples.

### Requirements
I use conda to manage virtual environments so you will need [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or just install all the packages manually).
To install the dependencies with conda use:

    conda env create -f environment.yml

### Running

    python dqn_cartpole.py

With the default hyper-params it should start learning at about 13k frames and it should reach R100 of 195 at about 33k. 