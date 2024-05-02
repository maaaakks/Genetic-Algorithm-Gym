import torch
import gymnasium as gym
import classes.GeneticAlgorithm as GA
import classes.Graphs as graph

class Environment:
    def __init__(self, env_name, render_mode, population_size, generations, mutation_rate, min_mutation_rate, selection_rate):
        self.env = gym.make(env_name, render_mode=render_mode) 
        self.GA = GA.GeneticAlgorithm(self.env, population_size, selection_rate)
        self.render_mode = render_mode
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.population_size = population_size
        #self.fitness_max_history = []
        self.fitness_average_history = []
        if self.render_mode == 'rgb_array': self.graph = graph.Graph(); self.line, self.ax, self.fig = self.graph.init_graph()
        
    def select_action(self, individual, state): # select action based on the state using the torch network
        state = torch.from_numpy(state).float()
        output = individual(state)
        return output.argmax().item()
        
    def run_individual(self, individual, generation): # run the individual in the environment
        score = 0
        state, _ = self.env.reset() # add seed=generation to use static seed for each generation
        while True:
            action = self.select_action(individual, state)
            state, reward, terminated, truncated, _ = self.env.step(action)
            score += reward
            if terminated or truncated: return score
          
    def run_simulation(self): # run the simulation
        self.GA.load_population()     
        
        for generation in range(self.generations): # for each generation
            fitness_scores = []
            
            for individual in self.GA.population: # for each individual
                score = self.run_individual(individual, generation)
                fitness_scores.append(score)
                if self.render_mode == 'human': print("Individual score: ", score) # Display the individual score if the render mode is human
            # end individual    
            
            # Display the data on the graph and the terminal
            #self.fitness_max_history.append(max(fitness_scores)) # max_fitness_score if you need the information
            self.fitness_average_history.append(sum(fitness_scores) / len(fitness_scores))
            print(f"Generation {generation} - Average: {round (sum(fitness_scores) / len(fitness_scores), 2)} - Best: {round(max(fitness_scores), 2)} - Mutation_rate: {round(self.mutation_rate,2)}")
            
            # Evolve the population and save the best individual
            self.GA.evolve(fitness_scores, self.mutation_rate)
            self.mutation_rate = max(self.mutation_rate *0.99, self.min_mutation_rate)
            
            # Display the data on the graph
            if self.render_mode == 'rgb_array': self.graph.update_graph(self.fitness_average_history)     
        if self.render_mode == 'rgb_array': graph.plt.ioff(); graph.plt.show()