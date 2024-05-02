"""
Description : Gym genetic algorithm working with a lot of environments with discrete actions
Gymnasium : https://gymnasium.farama.org/index.html
Author : https://github.com/maaaakks

Cheat Sheet Environments :
    - LunarLander-v2
    - CartPole-v1
    - MountainCar-v0
"""

import classes.EnvironmentRunner as env
import sys

# Get the name of the environment to use from the command line arguments, or default
args = sys.argv
arg = str(args[1]) if len(args) > 1 else 'CartPole-v1'

environment = env.Environment(
    env_name=arg,
    render_mode="rgb_array", # rgb_array or human
    
    population_size=100,
    generations=2500,
    
    # Hyperparameters : 
    # Adjust these to get better results for each environment and training phase
    
    mutation_rate=0,
    min_mutation_rate=0.05,
    selection_rate=0.1
    )

environment.run_simulation()
