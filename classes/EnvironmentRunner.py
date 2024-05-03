import torch
import gymnasium as gym
import classes.GeneticAlgorithm as GA
import classes.Graphs as graph

class Environment:
    def __init__(self, env_name, render_mode, save_data, population_size, generations, mutation_rate, min_mutation_rate, selection_rate):
        self.env = gym.make(env_name, render_mode=render_mode) 
        self.GA = GA.GeneticAlgorithm(self.env, population_size, selection_rate)
        
        self.render_mode = render_mode
        self.save_data = save_data
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.min_mutation_rate = min_mutation_rate
        
        if self.save_data:self.graph = graph.Graph(env_name)
    
    
    def select_action(self, individual, state): # select action based on the state using the torch network
        state = torch.from_numpy(state).float()
        output = individual(state)
        return output.argmax().item()
        
        
    def run_individual(self, individual, generation): # run the individual in the environment
        score = 0
        frame = 0
        state, _ = self.env.reset() # add seed=generation to use static seed for each generation
        
        while True:
            action = self.select_action(individual, state)
            state, reward, terminated, truncated, _ = self.env.step(action)
            score += reward
            frame += 1

            if terminated or truncated : return score 
        
        
    def run_simulation(self): # run the simulation
        self.GA.load_population()
        fitness_average_history = []    
         
        for generation in range(self.generations): # for each generation
            fitness_scores = []
            for individual in self.GA.population: # for each individual
                score = self.run_individual(individual, generation)
                fitness_scores.append(score)
            
            average_fitness = round(sum(fitness_scores) / len(fitness_scores),2)
            best_fitness = round(max(fitness_scores),2)
            fitness_average_history.append(average_fitness)
            
            self.GA.evolve(fitness_scores, self.mutation_rate)
            self.mutation_rate = max(self.mutation_rate *0.99, self.min_mutation_rate)
            
            print(f"Generation {generation} - Average: {average_fitness} - Best: {best_fitness} - Mutation_rate: {round(self.mutation_rate,2)}")
            if self.save_data:self.graph.log_metrics(average_fitness, best_fitness, generation)
        self.graph.close()