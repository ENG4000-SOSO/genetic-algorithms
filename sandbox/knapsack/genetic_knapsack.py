import random
from knapsack_generator import generate_knapsack


# Define the genetic algorithm parameters
POPULATION_SIZE = 100      # Number of individuals in each generation
GENERATIONS = 100          # Number of generations to evolve
CROSSOVER_RATE = 0.5       # Probability of crossover
MUTATION_RATE = 0.001       # Probability of mutation per bit

# # Knapsack problem parameters
# weights = [2, 3, 4, 5]     # Example weights
# values = [3, 4, 5, 6]      # Example values
# capacity = 5               # Knapsack capacity
# num_items = len(weights)
_n = 1_000
_capacity = _n // 5
_weights, _values = generate_knapsack(_n)
# num_items = n

# Fitness function
def fitness(individual, weights, values, capacity, num_items):
    total_weight = sum(individual[i] * weights[i] for i in range(num_items))
    total_value = sum(individual[i] * values[i] for i in range(num_items))
    # Return the value if within capacity; otherwise, penalize
    return total_value if total_weight <= capacity else 0

# Generate initial population
def generate_population(population_size, num_items):
    population = []

    for _ in range(population_size):
        individual = [0] * num_items
        number_of_ones = random.randint(1, 5)
        random_indexes = random.sample(range(num_items), number_of_ones)

        for index in random_indexes:
            individual[index] = 1

        population.append(individual)

    return population
    # return [[random.randint(0, 1) for _ in range(num_items)] for _ in range(size)]

# Selection: Roulette wheel selection based on fitness
def select(population, weights, values, capacity, n):
    fitnesses = [fitness(ind, weights, values, capacity, n) for ind in population]
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population)  # Fallback in case all fitnesses are 0
    selection_probabilities = [f / total_fitness for f in fitnesses]
    return population[random.choices(range(len(population)), weights=selection_probabilities, k=1)[0]]

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
def genetic_algorithm(weights, values, capacity):
    n = len(weights)
    population = generate_population(POPULATION_SIZE, n)
    best_solution = None
    best_fitness = 0

    for generation in range(GENERATIONS):
        new_population = []
        for _ in range(POPULATION_SIZE):
            # Selection
            parent1 = select(population, weights, values, capacity, n)
            parent2 = select(population, weights, values, capacity, n)
            # Crossover
            offspring = crossover(parent1, parent2, n)
            # Mutation
            offspring = mutate(offspring)
            # Add to new population
            new_population.append(offspring)
        
        # Update population and best solution
        population = new_population
        for individual in population:
            ind_fitness = fitness(individual, weights, values, capacity, n)
            if ind_fitness > best_fitness:
                best_fitness = ind_fitness
                best_solution = individual
        
        # Print progress every 10 generations
        # if generation % 10 == 0:
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness

# Run the genetic algorithm
# solution, max_value = genetic_algorithm(_weights, _values, _capacity)
# print("Best solution found:", solution)
# print("Maximum value achieved:", max_value)
