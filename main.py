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

args = sys.argv
arg = str(args[1]) if len(args) > 1 else 'CartPole-v1'

environment = env.Environment(
    env_name=arg,
    render_mode="human", # rgb_array or human
    save_data = False, # set to True if you want to use tensorboard
    
    population_size=100,
    generations=30000,
    
    mutation_rate=0,
    min_mutation_rate=0.0, 
    selection_rate=0.5
    )

environment.run_simulation()
