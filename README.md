# Generic Genetic Algorithm for Gymnasium Environments

<p align="center">
  <img src="images/illustration.JPG" alt="illustration" width="50%">
</p>

## Project Overview
This project applies a simple genetic algorithm to optimize solutions in various Gymnasium game environments. 
The genetic algorithm iteratively adjusts neural networks to enhance performance in games.

The genetic algorithm showcased notable limitations in efficiency, primarily due to constraints within the gym environment preventing parallel execution of the population's runs, its utility remained significant as it presents an intriguing way for exploring Machine Learning.
Its utilization underscores the vast potential for enhancing and optimizing genetic algorithms, facilitating faster capitalization on the discovery of novel features.

Tested on CartPole-v1, MountainCar-v0, and LunarLander-v2, but adaptable to most games with discrete actions and simple observation spaces.
For games with continuous actions and/or different observation structures, adjustments are required in both the variable definitions and the neural network function activations.
https://gymnasium.farama.org/

<div>
  <img src="images/mountainCarGif.gif" alt="MountainCarGif" width="32%">
  <img src="images/lunarLandingGif.gif" alt="LunarLandingGif" width="32%">
  <img src="images/cartPoleGif.gif" alt="CartPoleGif" width="32%">   
</div>

## Fields of Improvement
Significant enhancements can be made to the genetic algorithm:

- Crossover Algorithm: Integrate a more advanced crossover method that considers the fitness scores of parents. This approach should aim to selectively propagate superior traits.
- Mutation Algorithm: Implement a more nuanced mutation strategy. Adjust mutation rates dynamically to balance exploration in the initial phases and refinement in later phases.
- Hyperparameter Adjustment: Develop a method to dynamically adjust hyperparameters as the generations progress. Currently, the mutation rate is modified by a static factor; exploring adaptive adjustments could lead to improved convergence and performance of the algorithm.
- Neural Network: Reflection and testing need to be conducted on the neural network to optimize the architecture and the activation function used for these environments.

## Repository Contents
- `classes/`: Containing classes for neural network, environment runner, and genetic algorithms.
- `models/`: Saved pre-trained models for different environments.
- `runs/`: Saved running/training data
- Main script for running simulations and visualizing results.

## Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/maaaakks/genetic-algorithm-gym
   cd genetic-algorithm-gym
   
2. **Install dependencies:** 
    ```bash
    pip install torch gymnasium

3. **Run the simulation:** 
    Execute the simulation for a specific environment by providing its identifier as a command-line argument. You can also provide an optional second argument to specify the path to a neural network configuration YAML file. If no configuration file is specified, `nn_config_default_simple.yaml` will be used.
    Pre-trained models are available in the "models" folder and can be deleted to initiate training from scratch.
    ```bash
    # Run with default environment (CartPole-v1) and default NN config (nn_config_default_simple.yaml)
    python main.py

    # Run with a specific environment and default NN config
    python main.py LunarLander-v2

    # Run with a specific environment and a custom NN config
    python main.py MountainCar-v0 nn_config_lunarlander_test.yaml 
    # (assuming nn_config_lunarlander_test.yaml or your custom file exists)
    ```
    
### Environment Setup
1. **Initialization:**
   - The environment (e.g., CartPole-v1) is initialized with specific settings, such as :
        - rendering mode,
        - population size,
        - number of generations
        - hyperparameters like :
            - mutation rate,
            - minimun mutation rate,
            - selection rate.
    - Adjust these settings to suit your needs. Currently, the mutation rate is set to 0 to exploit the model after training.
    - The mutation rate is reduced by a factor of 0.99 after each generation to facilitate model convergence but can't go lower than the minimun mutation rate.

### Neural Network Configuration
The architecture of the neural network used by the agent is now highly configurable via a YAML file. By default, `main.py` uses `nn_config_default_simple.yaml` (a simple 2-layer network with 16 neurons each), but you can specify a custom configuration file as a command-line argument (see "Run the simulation" section).

The configuration file defines the hidden layers and the activation function for the output layer. The input layer's size is automatically determined by the environment's observation space, and the output layer's neuron count is determined by the environment's action space.

**YAML Structure:**
An example structure for a configuration file (e.g., `nn_config_default_simple.yaml` or your own `my_custom_nn.yaml`):
```yaml
# Example: nn_config_default_simple.yaml
neural_network:
  hidden_layers:
    - neurons: 16  # Number of neurons in the first hidden layer
      activation: 'ReLU'  # Activation function for the first hidden layer
    - neurons: 16  # Number of neurons in the second hidden layer
      activation: 'ReLU'  # Activation function for the second hidden layer
    # Add more hidden layers as needed
    # - neurons: N
    #   activation: 'FunctionName'
  output_layer:
    activation: 'Linear' # Activation for the output layer.
                         # Typically 'Linear' for discrete action spaces using argmax.
                         # Other PyTorch nn module names like 'Tanh', 'Sigmoid' can be used.
```

**Key Points:**
-   **`neural_network`**: The root key for network configuration.
-   **`hidden_layers`**: A list where each item defines a hidden layer with:
    -   `neurons`: The number of neurons in that layer.
    -   `activation`: The name of the PyTorch activation function (e.g., `ReLU`, `Tanh`, `Sigmoid`, `LeakyReLU`). If 'Linear' is specified, no explicit activation function is applied after that layer's linear transformation.
-   **`output_layer`**: Defines the activation for the output layer.
    -   `activation`: Typically 'Linear' for classification/discrete action spaces where `argmax` is used on the raw scores. For continuous actions bounded, e.g., in `[-1, 1]`, `Tanh` might be appropriate.

You can create your own `*.yaml` files with custom architectures (e.g., more layers, different neuron counts, different activation functions) and pass the filename to `main.py`. This allows for flexible experimentation with network structures without modifying the core Python code.

The old system of "simple" (`<input - 16 - 16 - output>`) and "complex" (`<input - 64 - 64 - 64 - output>`) architectures has been replaced by this YAML-based configuration. Example configurations like `nn_config_cartpole_test.yaml` (20-20 neurons) and `nn_config_lunarlander_test.yaml` (64-64-64 neurons) are provided as starting points.

### Genetic Algorithm Configuration
The behavior of the genetic algorithm can be fine-tuned through the `genetic_algorithm` section in your YAML configuration file (e.g., `nn_config_default_simple.yaml` or a custom file specified as a command-line argument). This allows you to experiment with different strategies for selection, crossover, and mutation to optimize learning for various environments, which can be particularly useful for balancing exploration and exploitation.

1.  **Population Initialization:**
    *   A population of neural networks (individuals) is generated.
    *   Each individual in the population is evaluated in the environment to determine its score (the total reward obtained during the environment simulation).
    *   The reward is based on the Gym Environment, without any addition or removal.

2.  **Evolution Process:**
    The core evolution loop involves selection, crossover, and mutation. These operations are now configurable:

    #### Selection
    Determines how individuals are chosen from the current population to become parents for the next generation. The `selection_rate` parameter (set in `main.py` or through its general settings) still defines the proportion of the population that will become parents.
    *   **`truncation`** (Default): Selects the top N% of the population based directly on their fitness scores. This is the original method.
    *   **`tournament`**: Conducts "tournaments" where a small group of individuals (`tournament_size`) are randomly chosen from the population, and the fittest individual from that group is selected as a parent. This process is repeated until the desired number of parents is chosen.
    *   **`rank_based`**: Selects individuals based on their rank (order when sorted by fitness) rather than their absolute fitness scores. This can help maintain diversity and prevent premature convergence, especially if a few individuals have vastly superior fitness scores.

    *YAML Example for Selection:*
    ```yaml
    # In your nn_config_xxx.yaml file
    genetic_algorithm:
      selection:
        type: 'tournament'  # Options: 'truncation', 'tournament', 'rank_based'
        tournament_size: 3    # Used if type is 'tournament'
      # selection_rate is a general parameter, not set here.
    ```

    #### Crossover
    Defines how genetic material (network weights and biases) from two selected parents is combined to create an offspring.
    *   **`average`** (Default): The child's parameters (weights and biases) are the element-wise average of the corresponding parameters from the two parents. This is the original method.
    *   **`uniform`**: Each individual parameter in the child's network is randomly chosen from one of the two parents with a certain probability (`mixing_probability`). For example, if `mixing_probability` is 0.5, each gene has a 50% chance of coming from parent1 and a 50% chance from parent2.
    *   **`one_point`**: All parameters from both parents are flattened into a single sequence. A single crossover point is randomly chosen in this sequence. The child inherits parameters from the first parent up to this point and from the second parent from this point onwards.

    *YAML Example for Crossover:*
    ```yaml
    genetic_algorithm:
      # ... (selection config)
      crossover:
        type: 'uniform'     # Options: 'average', 'uniform', 'one_point'
        mixing_probability: 0.5 # Used if type is 'uniform'
    ```

    #### Mutation
    Introduces random changes in an individual's parameters to maintain genetic diversity and explore new solutions in the search space.
    *   **`per_gene_mutation_prob`**: This is the probability (e.g., 0.01 for 1%, 0.05 for 5%) that each individual weight or bias ("gene") within a selected child network will undergo mutation.
    *   **`mutation_strength`**: This determines the magnitude of the random change (Gaussian noise) applied to a gene if it's selected for mutation. This strength is dynamically controlled by the `mutation_rate` parameter set in `main.py`'s `Environment` arguments (e.g., `mutation_rate=0.1`). This `mutation_rate` typically decays over generations (e.g., `self.mutation_rate = max(self.mutation_rate *0.99, self.min_mutation_rate)` in `EnvironmentRunner.py`), allowing for larger exploration steps early on and finer adjustments later.

    *YAML Example for Mutation:*
    ```yaml
    genetic_algorithm:
      # ... (selection and crossover config)
      mutation:
        per_gene_mutation_prob: 0.05 
        # mutation_strength is dynamically adjusted based on main.py's 
        # Environment 'mutation_rate' parameter.
    ```

3.  **Simulation Execution:**
    *   The simulation runs through the specified number of generations, with each generation applying the configured selection, crossover, and mutation steps.

4.  **Saving and Loading Populations:**
   - **Saving:** Allows the saving of the current population's state after each generation into a PyTorch file, named after the environment (e.g., `<environment_id>.pth`). This ensures that progress is not lost and can be reviewed or continued later.
   - **Loading:** Retrieves the saved model from the models directory, enabling the continuation or analysis of previously trained populations. If no saved model is found, a new population is initialized from scratch.

   
### Visualization and Analysis
- Fitness scores are collected and analyzed to track progress over generations in the runs folder.
- To visualize the progress using TensorBoard, execute the command `tensorboard --logdir=runs/` from a terminal at the root of the script.


## Results
After training, the genetic algorithm demonstrates remarkable performance.
population size = 100

### CartPole-v1
- Trained for ~20 generations.
- Perfect score (500) on every individual of populations over multiple generations.

<p align="center">
   <img src="images/cartPole_graph.JPG" alt="illustration">
   <img src="images/cartPoleGif.gif" alt="illustration">
</p>

### MountainCar-v0
- Trained for ~500 generations with a high mutation rate and low selection rate and static seed for each generation to prevent luck based selection.
- Average score higher than ~110 over multiple generations with a maximum score at -83 for the best individuals of each generations.
- Highly depend on the spawn location.

<p align="center">
   <img src="images/mountainCar_graph.JPG" alt="illustration">
   <img src="images/mountainCarGif.gif" alt="illustration">
</p>

### LunarLander-v2
- trained ~500 generations with high mutation rate, low selection rate and static seed for each generation to prevent luck based selection.
- Average score ~225 over multiple generations with a maximum score at ~325 for the best individual.
- Highly depend on the spawn location.

<p align="center">
   <img src="images/lunarLander_graph.JPG" alt="illustration">
   <img src="images/lunarLandingGif.gif" alt="illustration">
</p>

## Contribution
Feel free to fork this project, submit pull requests, or propose new features or environments for optimization.