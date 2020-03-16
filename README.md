# Overview

![alt text](https://www2.minijuegosgratis.com/v3/games/thumbnails/205296_1.jpg "sonic")

The main objective of the project is to study how to train a neural network to learn to play classic games.

Artificial intelligence (AI) in video games is a longstanding research area. It studies how to use AI technologies to achieve human-level performance when playing
games. More generally, it studies the complex interactions between agents and game environments. Various games provide interesting and complex problems for agents to solve, making video games perfect environments for AI research. On the other side, AI has been helping games to become better in the way we play, understand and design them.

# Reinforcement learning:

Reinforcement learning is a kind of machine learning methods where agents learn the optimal policy by trial and error. By interacting with the environment, RL can be successfully applied to sequential decision-making tasks. 

Considering a discounted episodic Markov decision process (MDP), the agent chooses an action at according to the policy at state. The environment receives the action, produces a reward and transfers to the next state.

![alt text](https://www.damiankolmas.com/images/MDP_framework_v1.png "MDP")


The process continues until the agent reaches a terminal state or a maximum time step. The objective is to maximize the expected discounted cumulative rewards.

# Gym retro:

Gym Retro (https://github.com/openai/retro) is a relatively new (released on May 25, 2018) Python library released by OpenAI (https://blog.openai.com/gym-retro/) as a research platform for developing reinforcement learning algorithms for game playing.

It facilitates the user to create a game environment and connect it to a neural network.

# NEAT:

NeuroEvolution of Augmenting Topologies (NEAT) is a genetic algorithm (GA) for the generation of evolving artificial neural networks (a neuroevolution technique) developed by Ken Stanley in 2002 while at The University of Texas at Austin. It alters both the weighting parameters and structures of networks, attempting to find a balance between the fitness of evolved solutions and their diversity. It is based on applying three key techniques: tracking genes with history markers to allow crossover among topologies, applying speciation (the evolution of species) to preserve innovations, and developing topologies incrementally from simple initial structures ("complexifying").

Traditionally a neural network topology is chosen by a human experimenter, and effective connection weight values are learned through a training procedure. This yields a situation whereby a trial and error process may be necessary in order to determine an appropriate topology. NEAT is an example of a topology and weight evolving artificial neural network (TWEANN) which attempts to simultaneously learn weight values and an appropriate topology for a neural network.


# Stable Baselines & Tensorflow comparision:

We also will make tests with stable baselines improved implementations of Reinforcement Learning (RL) algorithms and Tensorflow (It is a symbolic math library, and is also used for machine learning applications such as neural networks It is used for both research and production at Google.)
