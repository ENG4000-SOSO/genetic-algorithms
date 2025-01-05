import random
from typing import List, Set, Dict


# Define the genetic algorithm parameters
POPULATION_SIZE = 100      # Number of individuals in each generation
GENERATIONS = 100          # Number of generations to evolve
CROSSOVER_RATE = 0.5       # Probability of crossover
MUTATION_RATE = 0.001       # Probability of mutation per bit

# Generate initial population
def generate_population(population_size, num_items):
    population = []

    for _ in range(population_size):
        individual = [0] * num_items
        number_of_ones = random.randint(1, 2)
        random_indexes = random.sample(range(num_items), number_of_ones)

        for index in random_indexes:
            individual[index] = 1

        population.append(individual)

    return population

# Fitness function
def fitness(individual: List[int], conflicts: List[Set[int]], priorities: List[int]):
    one_indexes = [i for i, x in enumerate(individual) if x == 1]

    one_indexes_set = set(one_indexes)

    value = 0

    for index in one_indexes:
        duplicates = conflicts[index].intersection(one_indexes_set)
        value += priorities[index]
        if len(duplicates) > 0:
            return 0

    return value

# Selection: Roulette wheel selection based on fitness
def select(population: List[List[int]], conflicts: List[Set[int]], priorities: List[int]):
    fitnesses = [fitness(individual, conflicts, priorities) for individual in population]

    total_fitness = sum(fitnesses)

    if total_fitness == 0:
        return random.choice(population)  # Fallback in case all fitnesses are 0

    selection_probabilities = [f / total_fitness for f in fitnesses]

    selected_individual = random.choices(
        population,
        weights=selection_probabilities,
        k=1
    )[0]

    return selected_individual

# Crossover: Single-point crossover
def crossover(parent1, parent2, num_items):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, num_items - 1)
        return parent1[:point] + parent2[point:]
    return parent1[:]

# Mutation: Bit flip mutation
def mutate(individual):
    return [bit if random.random() > MUTATION_RATE else 1 - bit for bit in individual]

# Main genetic algorithm
def genetic_algorithm(conflicts: List[Set[int]], priorities: List[int]):
    n = len(conflicts)
    population = generate_population(POPULATION_SIZE, n)
    best_solution = None
    best_fitness = 0

    for generation in range(GENERATIONS):
        new_population = []
        for _ in range(POPULATION_SIZE):
            # Selection
            parent1 = select(population, conflicts, priorities)
            parent2 = select(population, conflicts, priorities)
            # Crossover
            offspring = crossover(parent1, parent2, n)
            # Mutation
            offspring = mutate(offspring)
            # Add to new population
            new_population.append(offspring)
        
        # Update population and best solution
        population = new_population
        for individual in population:
            ind_fitness = fitness(individual, conflicts, priorities)
            if ind_fitness > best_fitness:
                best_fitness = ind_fitness
                best_solution = individual
        
        # Print progress every 10 generations
        # if generation % 10 == 0:
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness
