"""
Description : Gym genetic algorithm working with a lot of environments with discrete actions
Gymnasium : https://gymnasium.farama.org/
Author : https://github.com/maaaakks

Cheat Sheet Environments :
    - LunarLander-v2
    - CartPole-v1
    - MountainCar-v0
"""

import classes.EnvironmentRunner as env
import sys
import yaml

args = sys.argv
env_arg = str(args[1]) if len(args) > 1 else 'CartPole-v1'
config_file_arg = str(args[2]) if len(args) > 2 else 'nn_config_default_simple.yaml' # New

print(f"Using environment: {env_arg}") # Added for clarity
print(f"Using NN config file: {config_file_arg}") # Added for clarity

# Load configuration from the specified config file
try:
    with open(config_file_arg, 'r') as f: # Use config_file_arg
        app_config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Configuration file {config_file_arg} not found. Exiting.")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing YAML configuration file {config_file_arg}: {e}. Exiting.")
    sys.exit(1)

network_config = app_config.get('neural_network')
if not network_config:
    print(f"Error: 'neural_network' section not found in {config_file_arg}. Exiting.")
    sys.exit(1)

environment = env.Environment(
    env_name=env_arg, # Use env_arg
    render_mode="human", # rgb_array or human
    save_data = False, # set to True if you want to use tensorboard
    network_config=network_config, # Add this
    population_size=100,
    generations=30000,
    
    mutation_rate=0, # Initial mutation rate, consider moving to config
    min_mutation_rate=0.0, 
    selection_rate=0.5
    )

environment.run_simulation()
