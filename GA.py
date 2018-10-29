import numpy as np
import pandas as pd
from tqdm import tqdm


class GeneticAlgorithm:

    def __init__(self, population_size, p_crossover, p_mutation, n_generations, n_genes, n_outputs, variable_range):
        self.population_size = population_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.n_generations = n_generations
        self.n_genes = n_genes
        self.n_outputs = n_outputs
        self.variable_range = variable_range

        self.evolve()

    def initialize(self):
        """
        Initializes the binary population randomly according to size and genes
        # Arguments:
            population_size: Size of population to be initiated
            n_genes: The number of genes of the algorithm

        # Returns:
            population: A randomly created population
        """

        return np.random.choice([0, 1], size=(self.population_size, self.n_genes))

    def decode(self, chromosome):
        """
        Decodes the chromosome from a binary sequence to a real-valued number
        # Arguments:
            chromosome: Binary array
            n_outputs: Number of outputs to optimize
            variable_range: Range of the outputs

        # Returns:
            decoded_chromosome: Translated chromosome to real-value
        """
        k_bits = np.round(len(chromosome)/self.n_outputs)

        x = np.zeros(self.n_outputs, dtype=np.int32)
        for i in range(self.n_outputs)[:-1]:
            for j in range(k_bits):
                x[i+1] = x[i+1] + chromosome[j+(i*k_bits)]*2 ^ (-j)

            x[i+1] = (-self.variable_range + 2*self.variable_range*x[i+1]/(1-2 ^ k_bits))

        decoded_chromosome = x

        return decoded_chromosome

    def select(self):
        """
        Performs a tournament of random individuals and returns the fittest in the subset of the population
        # Arguments:

        # Returns:
        """
        return []

    def breed(self):
        """

        # Arguments:
        # Returns:
        """
        return []

    def replace(self):
        """

        # Arguments:
        # Returns:
        """
        return []

    def mutate(self):
        """

        # Arguments:
        # Returns:
        """
        return []

    def evaluate(self, individual):
        """
        Evaluates a single chromosome on a specified fitness-function to optimize towards
        # Arguments:
            individual: A single decoded chromosome

        # Returns:
            fitness: A fitness score for that specific individual
        """

        return individual[0] ^ 2 - individual[1] ^ 3 + 25

    def stop(self):
        """

        # Arguments:
        # Returns:
        """
        return False

    def evolve(self):
        """
        Main-function for evolving the population.
        1. Initialization. 2. Selection. 3. Tournament. 4. Crossover. 5. Mutation. 6. Insert.
        # Arguments:
        # Returns:
        """
        fitness = np.zeros(self.population_size, dtype=np.float32)
        individual = np.zeros((self.population_size, self.n_outputs), dtype=np.float32)

        population = self.initialize()
        for iGeneration in tqdm(range(self.n_generations)):
            print('Generation: ' + str(iGeneration))

            max_fitness = 0.0
            for i in range(self.population_size):
                chromosome = population[i, :]
                decoded_chromosome = self.decode(chromosome)
                individual[i, :] = decoded_chromosome
                fitness[i] = self.evaluate(decoded_chromosome)

                if fitness[i] > max_fitness:
                    max_fitness = fitness[i]
                    best_individual = chromosome
                    x_best = decoded_chromosome

            temp_population = population

            for j in range(self.)


        minimal_value = 1 / max_fitness

        return 0.0


if __name__ == '__main__':
    seed = 12
    population_size = 5
    p_crossover = 0.8
    p_mutation = 0.025
    n_generations = 100
    n_genes = 50
    n_outputs = 2
    variable_range = 5.0

    GeneticAlgorithm(population_size,
                     p_crossover,
                     p_mutation,
                     n_generations,
                     n_genes,
                     n_outputs,
                     variable_range)
