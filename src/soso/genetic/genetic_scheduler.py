'''
The genetic algorithm aspect of the SOSO solution to the job-satellite
scheduling project.

The algorithm starts with a population. Each individual of the population is an
instance of the scheduling problem: a set of jobs and a set of satellites. The
algorithm proceeds in the following steps:

1. Two individuals are selected from the population. Individuals with a higher
`fitness` have a higher chance of being selected.

2. The selected individuals are possibly `crossed-over`, as in parts of their
instances are mixed with each other to make a new `offspring` instance. The
decision to perform this crossover depends on the `crossover rate`, which is a
hyperparameter.

3. The offspring instance is mutated by restricting or un-restricting some jobs
from being scheduled. The amount of mutation depends on the `mutation rate`,
which is a hyperparameter.

4. Repeat this selection, crossover, and mutation until a new population of
equal size is generated, which then replaces the original population.

5. Repeat this entire process for a certain number of `generations`, which is a
hyperparameter.

When computing the fitness of a solution, the more fit solutions should have:
  - more jobs scheduled,
  - more higher priority jobs scheduled, and
  - more equal usage of satellites and space resources.

Since each individual in the population is a problem *instance*, instead of
measuring the fitness of the problem instance, we measure the fitness of the
optimal solution to the instance, which we compute with a network flow
algorithm. Given the solution to an instance (set of jobs scheduled in timeslots
in the set of satellites), we can then compute metrics like number of jobs
scheduled, priorities of jobs scheduled, and variance of space resource use.
'''


from dataclasses import dataclass
import logging
import random
import time
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from skyfield.api import EarthSatellite, Loader, Timescale

from soso.job import Job
from soso.network_flow.edge_types import JobToSatelliteTimeSlotEdge
from soso.network_flow.network_flow_scheduler_improved import run_network_flow
from soso.outage_request import OutageRequest


POPULATION_SIZE = 30
'''
The number of problem instances to be considering at any given time.
'''

GENERATIONS = 20
'''
The number of iterations of the genetic algorithm.
'''

CROSSOVER_RATE = 0.05
'''
The crossover rate influences the chance of two selected individuals in a
population 'crossing over' to make a new individual.
'''

MUTATION_RATE = 0.001
'''
The mutation rate influences the chance of random mutations in individuals,
which means the restriction or un-restriction of random jobs from being
scheduled.
'''

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ProblemInstance:
    '''
    A representation of a problem instance for the SOSO scheduling problem.
    '''

    satellites: List[EarthSatellite]
    '''
    The list of satellites.
    '''

    jobs: List[Job]
    '''
    The list of jobs to be scheduled into satellites.
    '''

    outage_requests: List[OutageRequest]
    '''
    The list of outage requests to be scheduled into satellites.
    '''


@dataclass
class Individual:
    '''
    An individual in the population of the genetic algorithm.
    '''

    genome: List[Literal[0, 1]]
    '''
    The individual's genome.

    If `genome[i] == 0`, then the job at `problem_instance.jobs[i]` is
    restricted from being scheduled. Otherwise, if `genome[i] == 1` then the job
    at `problem_instance.jobs[i]` is not restricted and will be considered by
    the network flow scheduling algorithm.
    '''

    fitness: Optional[float] = None
    '''
    The fitness of the individual.

    For the calculation of fitness, see the function `fitness` below.
    '''


def generate_population(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    outage_requests: List[OutageRequest],
    ts: Timescale
) -> Tuple[ProblemInstance, List[Individual]]:
    '''
    Generates an initial population.

    Args:
        satellites: The list of satellites to be scheduled with jobs and outage
        requests.

        jobs: The list of jobs to be scheduled.

        outage_requests: The list of (non-negotiable) outage requests.

        ts: The Skyfield timescale being used to simulate events in the future.

    Returns:
        A tuple containing the problem instance (which has a list of satellites,
        a list of jobs, and a list of outage requests) and a list of individuals
        in the population.
    '''

    # Pack satellites, jobs, and outage requests into problem instance
    problem_instance = ProblemInstance(satellites, jobs, outage_requests)

    # Generate individuals in the population
    population: List[Individual] = []
    for _ in range(POPULATION_SIZE):
        # Each individual's genome is a list of 0's with a few 1's
        genome = [0] * len(jobs)
        number_of_ones = random.randint(1, 5)
        random_indexes = random.sample(range(len(jobs)), number_of_ones)

        for index in random_indexes:
            genome[index] = 1

        individual = Individual(genome)
        population.append(individual)

    return problem_instance, population


def fitness(
    genome: List[Literal[0, 1]],
    problem_instance: ProblemInstance,
    ts: Timescale,
    eph: Loader
) -> float:
    '''
    Calculates the fitness of an individual's genome.

    Args:
        genome: The genome to calculate the fitness of.

        problem_instance: The global problem instance.

        ts: The Skyfield timescale being used to simulate events in the future.

        eph: The Skyfield ephemeris data being used to perform astronomical
        calculations.

    Returns:
        The fitness of the genome as a floating point number.
    '''

    # Get all non-restricted jobs
    jobs = []
    for i, job in enumerate(problem_instance.jobs):
        if genome[i] == 1:
            jobs.append(job)
        else:
            pass

    # Run the network flow optimization scheduling algorithm to get the best
    # solution given non-restricted jobs
    solution = run_network_flow(
        problem_instance.satellites,
        jobs,
        problem_instance.outage_requests,
        ts,
        eph
    )

    # Calculate the variance of the amount of jobs scheduled in each satellite
    jobs_in_each_satellite = [
        len(job_to_timeslot_edges)
            for satellite, job_to_timeslot_edges in solution.items()
    ]
    variance = np.var(jobs_in_each_satellite)

    # Calculate the weighted sum of the priorities of all jobs
    total_job_priority_sum = sum(
        int(job.priority.value) for job in problem_instance.jobs
    )

    # Calculate the weighted sum of the priorities of all jobs that were
    # scheduled
    total_scheduled_job_priority_sum = sum(
        job_to_timeslot_edge.job.priority.value
            for satellite, job_to_timeslot_edges in solution.items()
                for job_to_timeslot_edge in job_to_timeslot_edges
    )

    if total_job_priority_sum == 0:
        # Returning 0 here to avoid division by zero exception
        return 0.0

    return float(
        (total_scheduled_job_priority_sum / total_job_priority_sum) - variance
    )


def select(population: List[Individual]) -> Individual:
    '''
    Selects an individual from the population where individuals with higher
    fitnesses have higher chances of being selected.

    Args:
        population: The population from which to select the individual.

    Returns:
        The selected individual.
    '''

    # Convert the list of individuals into a list of fitnesses
    fitnesses = [individual.fitness for individual in population]

    total_fitness = sum(fitnesses)

    # If the total fitness is 0, select randomly
    if (total_fitness == 0):
        return random.choice(population)

    # Select an individual where individuals with higher fitnesses have higher
    # chances of being selected
    selection_probabilities = [f / total_fitness for f in fitnesses]
    return population[
        random.choices(
            range(len(population)),
            weights=selection_probabilities,
            k=1
        )[0]
    ]


def crossover(
    parent1: Individual,
    parent2: Individual,
    problem_instance: ProblemInstance,
    ts: Timescale,
    eph: Loader
) -> Individual:
    '''
    Mix two individuals' genomes, potentially generating a new individual.

    The chance of a new individual being generated depends on the
    `CROSSOVER_RATE`. If a new individual is not generated, one of the parents
    is returned.

    Args:
        parent1: The first individual.

        parent2: The second individual.

        problem_instance: The global problem instance.

        ts: The Skyfield timescale being used to simulate events in the future.

        eph: The Skyfield ephemeris data being used to perform astronomical
        calculations.

    Returns:
        Either a new individual that is a combination of the two parents, or one
        of the parents, depending (randomly) on the crossover rate.
    '''

    # Randomly choose whether or not to crossover, depending on the crossover
    # rate
    if random.random() < CROSSOVER_RATE:
        # Randomly choose an index to slice the genomes
        n = len(parent1.genome)
        point = random.randint(1, n - 1)

        # Generate a new genome by mixing the first part of the first parent's
        # genome with the second part of the second parent's genome
        new_genome = parent1.genome[:point] + parent2.genome[point:]

        # Calculate the fitness of the new genome
        new_fitness = fitness(new_genome, problem_instance, ts, eph)

        return Individual(new_genome, new_fitness)

    # If not crossing over, just return the first parent
    return parent1


def mutate(
    individual: Individual,
    problem_instance: ProblemInstance,
    ts: Timescale,
    eph: Loader
) -> Individual:
    '''
    Mutate an individual's genome, potentially flipping some of its bits.

    The chance of a bits in a genome being flipped depends on the
    `MUTATION_RATE`.

    Args:
        individual: The individual whose genome is being mutated.

        problem_instance: The global problem instance.

        ts: The Skyfield timescale being used to simulate events in the future.

        eph: The Skyfield ephemeris data being used to perform astronomical
        calculations.

    Returns:
        A new individual with the mutated genome.
    '''

    for i in range(len(individual.genome)):
        # Randomly choose whether or not to flip a bit in the genome, depending
        # on the mutation rate
        if random.random() > MUTATION_RATE:
            if individual.genome[i] == 1:
                individual.genome[i] = 0
            else:
                individual.genome[i] = 1

    # Calculate the fitness of the mutated genome
    individual.fitness = fitness(individual.genome, problem_instance, ts, eph)

    return individual


def run_genetic_algorithm(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    outage_requests: List[OutageRequest],
    ts: Timescale,
    eph: Loader
) -> Dict[EarthSatellite, List[JobToSatelliteTimeSlotEdge]]:
    '''
    The main entry point to the genetic algorithm part of the SOSO scheduling
    algorithm.

    Args:
        satellites: The list of satellites to be scheduled with jobs and outage
        requests.

        jobs: The list of jobs to be scheduled.

        outage_requests: The list of (non-negotiable) outage requests.

        ts: The Skyfield timescale being used to simulate events in the future.

        eph: The Skyfield ephemeris data being used to perform astronomical
        calculations.

    Returns:
        A dictionary mapping each satellite to a list, where items in the list
        are representations of a job scheduled in a time slot. This is returned
        directly from the network flow algorithm.
    '''

    logger.info('Starting genetic algorithm')

    start_time = time.time()

    best_individual: Optional[Individual] = None
    '''
    The individual with the highest fitness score.
    '''

    best_fitness = 0
    '''
    The best individual's fitness score.
    '''

    problem_instance, population = generate_population(
        satellites,
        jobs,
        outage_requests,
        ts
    )

    for generation in range(GENERATIONS):
        logger.info(f'Generation {generation}')

        # Make sure all individuals have a fitness metric
        for individual in population:
            if not individual.fitness:
                individual.fitness = fitness(
                    individual.genome,
                    problem_instance,
                    ts,
                    eph
                )
            if not best_individual:
                # Make sure the best individual is not None
                best_individual = individual
                best_fitness = individual.fitness

        # Generate a new population by selecting parents, crossing over genomes,
        # and mutating genomes
        new_population: List[Individual] = []
        for _ in range(POPULATION_SIZE):
            parent1 = select(population)
            parent2 = select(population)
            offspring = crossover(parent1, parent2, problem_instance, ts, eph)
            mutated_offspring = mutate(offspring, problem_instance, ts, eph)
            new_population.append(mutated_offspring)

        # Replace the original population with the new population
        population = new_population

        # Update the best individual with the new population
        for individual in population:
            if individual.fitness > best_fitness:
                best_individual = individual
                best_fitness = individual.fitness

    end_time = time.time()

    logger.info(f'Genetic algorithm took {end_time-start_time} seconds')

    # Run the network flow algorithm one last time to get the solution to the
    # problem instance and the best individual's genome
    return run_network_flow(
        problem_instance.satellites,
        [
            job
                for i, job in enumerate(problem_instance.jobs)
                    if best_individual.genome[i] == 1
        ],
        problem_instance.outage_requests,
        ts,
        eph
    )
