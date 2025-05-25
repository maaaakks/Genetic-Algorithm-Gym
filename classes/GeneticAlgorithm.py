import classes.NeuralNetwork as NN
import torch, random, copy

class GeneticAlgorithm:
    def __init__(self, env, network_config, population_size, selection_rate):
        self.save_file = str("models/" + env.unwrapped.spec.id + ".pth")
        self.network_config = network_config # Store network_config
        self.population_size = population_size
        self.selection_rate = selection_rate
        
        # Get input_size and output_size from env
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n
        
        self.population = [NN.NeuralNetwork(input_size, output_size, self.network_config) for _ in range(population_size)]

    # Genetic selection functions
    def select(self, fitness_scores): # Select a proportion of the population to become parents based on fitness scores
        num_parents = int(len(self.population) * self.selection_rate) 
        parents = []
        sorted_population = sorted(zip(self.population, fitness_scores), key=lambda x: x[1], reverse=True)
        for i in range(num_parents):
            parents.append(sorted_population[i][0])
        return parents

    def crossover(self, parent1, parent2): # Simple crossover 50%
        child = copy.deepcopy(parent1)
        with torch.no_grad():
            for child_param, param1, param2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                child_param.data.copy_((param1.data + param2.data) / 2)
        return child

    def mutate(self, individual, mutation_rate): # Mutate weights of the individuals
        with torch.no_grad():
            for param in individual.parameters():
                if random.random() < mutation_rate:
                    noise = torch.randn_like(param) * mutation_rate
                    param.add_(noise)


    def evolve(self, fitness_scores, mutation_rate): # Evolve the population using Genetic selection functions
        parents = self.select(fitness_scores)
        next_population = []
        while len(next_population) < self.population_size:
            p1, p2 = random.sample(parents, 2)
            child = self.crossover(p1, p2)
            self.mutate(child, mutation_rate)
            next_population.append(child)
        self.population = next_population
        self.save_population(next_population) 


    # Saving and loading population
    def save_population(self, population):
        try:
            population_state = [individual.state_dict() for individual in population]
            torch.save(population_state, self.save_file)
        except:
            print("Failed to save population.")
            
    def load_population(self):
        try:
            
            population_state = torch.load(self.save_file)
            for individual, state in zip(self.population, population_state):
                individual.load_state_dict(state)
            print("Saved population found. Loading saved population.")
            
        except FileNotFoundError:
            print("No saved population found. Starting from scratch.")