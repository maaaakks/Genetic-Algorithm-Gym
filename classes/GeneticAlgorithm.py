import classes.NeuralNetwork as NN
import torch, random, copy

class GeneticAlgorithm:
    def __init__(self, env, network_config, ga_config, population_size, selection_rate):
        self.save_file = str("models/" + env.unwrapped.spec.id + ".pth")
        self.network_config = network_config # Store network_config
        
        # Store GA operator configurations
        self.selection_config = ga_config.get('selection', {'type': 'truncation'})
        self.crossover_config = ga_config.get('crossover', {'type': 'average'})
        self.mutation_config = ga_config.get('mutation', {}) # Default to empty, specific defaults handled in evolve
        
        self.population_size = population_size
        self.selection_rate = selection_rate # Retained as it's used by selection methods
        
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

    def tournament_select(self, fitness_scores, tournament_size=3):
        parents = []
        population_with_fitness = list(zip(self.population, fitness_scores))
        num_parents_to_select = int(len(self.population) * self.selection_rate)
        if num_parents_to_select == 0 and len(self.population) > 0 : # Ensure at least one parent if possible
             num_parents_to_select = 1
        
        if not population_with_fitness: # Handle empty population
            return []

        # Ensure tournament_size is not greater than the population size
        actual_tournament_size = min(tournament_size, len(population_with_fitness))
        if actual_tournament_size == 0: # if population is empty or tournament size is 0
            return []

        for _ in range(num_parents_to_select):
            # Ensure we can sample `actual_tournament_size` contenders
            if len(population_with_fitness) < actual_tournament_size:
                # Not enough individuals for a full tournament,
                # could either skip, or pick the best of the remaining, or sample with replacement.
                # For now, if not enough unique contenders, we might get fewer parents than desired.
                # Or, we can simply break if we can't form a full tournament
                if not population_with_fitness: # No more individuals left
                    break
                # If fewer individuals than tournament size, make them all contenders
                tournament_contenders = population_with_fitness
            else:
                tournament_contenders = random.sample(population_with_fitness, actual_tournament_size)
            
            winner = max(tournament_contenders, key=lambda x: x[1])[0]
            parents.append(winner)
        return parents

    def rank_based_select(self, fitness_scores):
        num_parents_to_select = int(len(self.population) * self.selection_rate)
        if num_parents_to_select == 0 and len(self.population) > 0:
            num_parents_to_select = 1
        
        if not self.population: # Handle empty population
            return []

        # Pair population with fitness and sort by fitness (descending: best fitness first)
        sorted_population_with_fitness = sorted(zip(self.population, fitness_scores), key=lambda x: x[1], reverse=True)
        
        # Extract sorted population (individuals only)
        sorted_population = [ind for ind, fit in sorted_population_with_fitness]

        num_individuals = len(sorted_population)
        if num_individuals == 0:
            return []

        # Weights: Best individual gets weight `num_individuals`, worst gets 1.
        weights = [num_individuals - i for i in range(num_individuals)]

        # Handle cases where weights might not be suitable for random.choices
        # (e.g., if num_individuals is 1, weights would be [1])
        # or if num_parents_to_select is 0
        if not weights or num_parents_to_select == 0:
            # If num_parents_to_select is 0, return empty list.
            if num_parents_to_select == 0:
                return []
            # If weights are empty (should not happen if num_individuals > 0) or population is small,
            # fall back to simple selection or return available individuals if fewer than num_parents_to_select.
            # This fallback might need refinement based on desired behavior for tiny populations.
            # For now, if sorted_population exists, sample from it.
            if sorted_population:
                return random.sample(sorted_population, min(num_parents_to_select, len(sorted_population)))
            else:
                return []

        # Select parents using weighted random choice.
        # random.choices allows for replacement by default, which is generally acceptable in GAs
        # as a good individual can be selected multiple times to be a parent.
        selected_parents = random.choices(sorted_population, weights=weights, k=num_parents_to_select)
        
        return selected_parents

    def crossover(self, parent1, parent2): # Simple crossover 50%
        child = copy.deepcopy(parent1)
        with torch.no_grad():
            for child_param, param1, param2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                child_param.data.copy_((param1.data + param2.data) / 2)
        return child

    def uniform_crossover(self, parent1, parent2, mixing_probability=0.5):
        child = copy.deepcopy(parent1)
        with torch.no_grad():
            for child_param, p1_param, p2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                if p1_param.data.shape == p2_param.data.shape:
                    mask = torch.rand_like(p1_param.data) < mixing_probability
                    # Ensure mask is float for multiplication if params are float, or bool if that's preferred and works.
                    # PyTorch typically handles bool masks correctly by converting them.
                    child_param.data.copy_(p1_param.data * mask.float() + p2_param.data * (1 - mask.float()))
                else:
                    # Fallback or error if shapes don't match, though they should for same model arch
                    child_param.data.copy_(p1_param.data) # Default to parent1's param
        return child

    def one_point_crossover(self, parent1, parent2):
        child1 = copy.deepcopy(parent1)
        with torch.no_grad():
            # 1. Flatten and gather info
            p1_params_flat_list = []
            p2_params_flat_list = []
            param_infos = [] 

            # Iterate over child's parameters to ensure correct order and inclusion
            child_params_dict = dict(child1.named_parameters())

            for p_name, p_child_tensor in child_params_dict.items():
                if p_child_tensor.requires_grad: 
                    # Ensure parent1 and parent2 have this parameter
                    if p_name not in dict(parent1.named_parameters()) or \
                       p_name not in dict(parent2.named_parameters()):
                        continue # Should not happen if architectures are identical

                    p1_tensor = dict(parent1.named_parameters())[p_name]
                    p2_tensor = dict(parent2.named_parameters())[p_name]
                    
                    param_infos.append({'name': p_name, 'shape': p_child_tensor.shape, 'numel': p_child_tensor.numel()})
                    p1_params_flat_list.append(p1_tensor.data.view(-1))
                    p2_params_flat_list.append(p2_tensor.data.view(-1))

            if not p1_params_flat_list: return child1 # No parameters to cross over

            p1_flat_tensor = torch.cat(p1_params_flat_list)
            p2_flat_tensor = torch.cat(p2_params_flat_list)
            total_params = p1_flat_tensor.numel()

            # 2. Crossover point
            if total_params <= 1: return child1 # Not enough params to crossover
            point = random.randint(1, total_params - 1)

            # 3. Create new flat tensor for child
            child_flat_tensor = torch.cat((p1_flat_tensor[:point], p2_flat_tensor[point:]))

            # 4. Unflatten and assign to child1
            current_pos = 0
            for info in param_infos:
                numel = info['numel']
                param_slice = child_flat_tensor[current_pos : current_pos + numel]
                # Access child's parameter tensor by name to assign data
                child_params_dict[info['name']].data.copy_(param_slice.view(info['shape']))
                current_pos += numel
        return child1

    def mutate(self, individual, per_gene_mutation_prob, mutation_strength): # Mutate weights of the individuals
        with torch.no_grad():
            for param in individual.parameters():
                if param.requires_grad: # Ensure we only mutate trainable parameters
                    # Create a random mask for each gene in the parameter tensor
                    mutation_mask = torch.rand_like(param.data) < per_gene_mutation_prob
                    # Generate Gaussian noise for the entire tensor
                    noise = torch.randn_like(param.data) * mutation_strength
                    # Apply noise only where the mask is True
                    param.data[mutation_mask] += noise[mutation_mask]


    def evolve(self, fitness_scores, mutation_rate): # Evolve the population using Genetic selection functions
        # Selection Logic
        selection_type = self.selection_config.get('type', 'truncation')
        if selection_type == 'tournament':
            tournament_size = self.selection_config.get('tournament_size', 3)
            parents = self.tournament_select(fitness_scores, tournament_size)
        elif selection_type == 'rank_based':
            parents = self.rank_based_select(fitness_scores)
        else: # Default or 'truncation'
            parents = self.select(fitness_scores)

        next_population = []
        while len(next_population) < self.population_size:
            if not parents:
                print("Warning: No parents selected, cannot evolve. Consider adjusting selection_rate or population size.")
                break
            
            if len(parents) == 1 and self.population_size >= 2:
                 p1 = parents[0]
                 p2 = parents[0] 
            elif len(parents) < 2:
                if parents:
                    p1 = parents[0]
                    p2 = parents[0]
                else:
                    break
            else:
                p1, p2 = random.sample(parents, 2)

            # Crossover Logic
            crossover_type = self.crossover_config.get('type', 'average')
            if crossover_type == 'uniform':
                mixing_prob = self.crossover_config.get('mixing_probability', 0.5)
                child = self.uniform_crossover(p1, p2, mixing_prob)
            elif crossover_type == 'one_point':
                child = self.one_point_crossover(p1, p2)
            else: # Default or 'average'
                child = self.crossover(p1, p2)
            
            # Mutation Logic
            per_gene_prob = self.mutation_config.get('per_gene_mutation_prob', 0.01) # Default value for per_gene_prob
            mutation_strength = mutation_rate # mutation_rate from evolve args is used as strength
            self.mutate(child, per_gene_prob, mutation_strength)
            
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