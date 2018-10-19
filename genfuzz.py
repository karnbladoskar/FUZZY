import numpy as np
import pandas as pd


def initialize(population_size, n_genes):
    """
    Initializes the binary population randomly according to size and genes
    # Arguments:
        population_size: Size of population to be initiated
        n_genes: The number of genes of the algorithm
        seed: Random seed for traceback

    # Returns:
        population: A randomly created population
    """

    return np.random.choice([0, 1], size=(population_size, n_genes))


def decode(chromosome, n_outputs, variable_range):
    """
    Decodes the chromosome from a binary sequence to a real-valued number
    # Arguments:
        chromosome: Binary array
        n_outputs: Number of outputs to optimize
        variable_range: Range of the outputs

    # Returns:
        decoded_chromosome:
    """
    k_bits = np.round(len(chromosome)/n_outputs)

    x = np.zeros(n_outputs, dtype=np.int32)
    for i in range(n_outputs)[:-1]:
        for j in range(k_bits):
            x[i+1] = x[i+1] + chromosome[j+(i*k_bits)]*2 ^ (-j)

        x[i+1] = (-variable_range + 2*variable_range*x[i+1]/(1-2 ^ k_bits))

    decoded_chromosome = x

    return decoded_chromosome


def select(self):
    """

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


def replace():
    """

    # Arguments:
    # Returns:
    """
    return []


def mutate():
    """

    # Arguments:
    # Returns:
    """
    return []


def evaluate(individual):
    """
    Evaluates a single chromosome on a specified fitness-function to optimize towards
    # Arguments:
        individual: A single decoded chromosome

    # Returns:
        fitness: A fitness score for that specific individual
    """

    return individual ^ 2 - individual ^ 3 + 25


def stop():
    """

    # Arguments:
    # Returns:
    """
    return False


def evolve(population_size, p_crossover, p_mutation, n_generations, n_genes, n_outputs, variable_range):
    """
    Main-function for evolving the population.
    1. Initialization. 2. Selection. 3. Tournament. 4. Crossover. 5. Mutation. 6. Insert.
    # Arguments:
    # Returns:
    """
    fitness = np.zeros(population_size, dtype=np.float32)
    population = initialize(population_size, n_genes)

    for iGeneration in range(n_generations):
        max_fitness = 0.0

        for i in range(population_size):
            chromosome = population[i, :]
            decoded_chromosome = decode(chromosome, n_outputs, variable_range)
            individual = decoded_chromosome[i]
            fitness[i] = evaluate(individual)

            if fitness[i] > max_fitness:
                max_fitness = fitness[i]
                best_individual = chromosome
                x_best = decoded_chromosome

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

    evolve(population_size,
           p_crossover,
           p_mutation,
           n_generations,
           n_genes,
           n_outputs,
           variable_range)