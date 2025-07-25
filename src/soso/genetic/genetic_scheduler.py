'''
The genetic algorithm aspect of the SOSO solution to the job-satellite
scheduling project.

The algorithm starts with a population. Each individual of the population is an
instance of the scheduling problem: a set of jobs and a set of satellites.
Individuals are represented by their `genome`.

An individual's genome is is a dictionary from satellites to a list of lists of
bits (0 or 1). This genome directly corresponds to the `satellite_intervals`
data structure, where satellites are mapped to lists of intervals, and each
interval contains a begin time, end time, and a list of jobs to be scheduled.

If `genome[satellite][i][j] == 1`, that means for this particular individual,
the `j`-th job in the `i`-th interval of `satellite` can be scheduled. If it is
0, the job cannot be scheduled. Therefore, an individual's genome effectively
encodes which jobs are restricted in each satellite.

The algorithm proceeds in the following steps:

1. Two individuals are selected from the population. Individuals with a higher
`fitness` have a higher chance of being selected.

2. The selected individuals are possibly `crossed-over`, as in parts of their
genomes are mixed with each other to make a new `offspring` instance. The
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
from pathlib import Path
import random
import time
from typing import List, Literal, Optional, Tuple

import numpy as np
from skyfield.api import EarthSatellite

from soso.debug import debug
from soso.interval_tree import SatelliteInterval, GroundStationPassInterval
from soso.job import Job
from soso.network_flow.edge_types import SatelliteToList
from soso.bin_packing.ground_station_bin_packing import \
    schedule_downlinks, \
    ScheduleUnit
from soso.network_flow.network_flow_scheduler import run_network_flow
from soso.outage_request import OutageRequest


POPULATION_SIZE = 20
'''
The number of problem instances to be considering at any given time.
'''

GENERATIONS = 25
'''
The number of iterations of the genetic algorithm.
'''

CROSSOVER_RATE = 0.25
'''
The crossover rate influences the chance of two selected individuals in a
population 'crossing over' to make a new individual.
'''

MUTATION_RATE = 0.01
'''
The mutation rate influences the chance of random mutations in individuals,
which means the restriction or un-restriction of random jobs from being
scheduled.
'''

INITIAL_ENABLE_PROBABILITY = 0.01
'''
The probability of enabling each job in an individual's genome when the initial
population is being generated.
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

    satellite_intervals: SatelliteToList[SatelliteInterval]
    '''
    The dictionary mapping satellites to intervals, where each interval has a
    beginning time, ending tme, and a list of jobs.
    '''

    ground_station_passes: SatelliteToList[GroundStationPassInterval]
    '''
    The dictionary mapping satellites to ground station passes, where each pass
    has a beginning time, ending tme, and a ground station.
    '''


@dataclass
class GeneticAlgorithmResult:
    '''
    A DTO containing the results of the genetic algorithm.
    '''

    result: SatelliteToList[ScheduleUnit]
    '''
    The result of the genetic algorithm.
    '''

    undownlinkable_jobs: List[Job]
    '''
    Jobs that could not be downlinked.
    '''

    optimized_out_jobs: List[Job]
    '''
    Jobs that were not scheduled as part of the optimization process.
    '''


def sigmoid(x):
    '''
    Sigmoid function to be used when normalizing metrics for fitness
    calculations.
    '''
    return 1 / (1 + np.exp(-x))


def optimize_schedule(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    satellite_intervals: SatelliteToList[SatelliteInterval],
    ground_station_passes: SatelliteToList[GroundStationPassInterval]
) -> GeneticAlgorithmResult:
    '''
    Generate an optimized schedule using network flow and bin packing
    algorithms.

    Args:
        satellites: The list of satellites.

        jobs: The list of jobs.

        satellite_intervals: The satellite intervals in which jobs can be
            scheduled.

        ground_station_passes: The ground station passes in which jobs can be
            downlinked.

    Returns:
        An optimized schedule where each satellite has a list of schedule units,
        containing a job, satellite timeslot, and a ground station pass.
    '''

    # Run the network flow optimization scheduling algorithm to get the best
    # solution given non-restricted jobs
    network_flow_result = run_network_flow(
        satellites,
        jobs,
        satellite_intervals
    )

    # Run the bin packing algorithm to assign ground station passes to each
    # satellite timeslot for downlinking
    bin_packing_result = schedule_downlinks(
        satellites,
        network_flow_result.job_to_sat_edges,
        ground_station_passes
    )

    # # Get the jobs that were not scheduled as part of both of the network flow
    # # and bin packing optimization algorithms
    # network_flow_unscheduled_jobs = set(network_flow_result.optimized_out_jobs)
    # bin_packing_unscheduled_out_jobs = set(bin_packing_result.infeasible_jobs)

    # optimized_out_jobs = list(
    #     network_flow_unscheduled_jobs.union(bin_packing_unscheduled_out_jobs)
    # )
    optimized_out_jobs = list(
        set(jobs).difference(
            set(
                schedule_unit.job
                    for sat, schedule_units in bin_packing_result.result.items()
                        for schedule_unit in schedule_units
            )
        )
    )

    return GeneticAlgorithmResult(
        bin_packing_result.result,
        bin_packing_result.impossible_jobs,
        optimized_out_jobs
    )


def get_satellite_intervals_from_genome(
    problem_instance: ProblemInstance,
    genome: SatelliteToList[List[Literal[0, 1]]]
) -> SatelliteToList[SatelliteInterval]:
    '''
    Gets satellite intervals that are enabled in the genome.

    Args:
        problem_instance: The problem instance of the algorithm.

        genome: The genome being used to filter satellite intervals.

    Returns:
        The dictionary mapping satellites to lists of intervals, where each
        interval has jobs enabled according to the genome.
    '''

    return {
        sat: [
            SatelliteInterval(
                interval.begin,
                interval.end,
                [job for job, bit in zip(interval.data, chromosome) if bit == 1]
            )
            for interval, chromosome in zip(intervals, chromosomes)
        ]
        for sat, (intervals, chromosomes) in zip(
            problem_instance.satellite_intervals.keys(),
            zip(problem_instance.satellite_intervals.values(), genome.values())
        )
    }


class Individual:
    '''
    An individual in the population of the genetic algorithm.
    '''

    __genome: SatelliteToList[List[Literal[0, 1]]]
    '''
    Private variable containing the individual's genome.
    '''

    __problem_instance: ProblemInstance
    '''
    Private variable containing the problem instance for the algorithm. This
    should be the same for every individual in the population (it just contains
    the full list of satellites, jobs, and satellite intervals).
    '''

    __fitness: Optional[float] = None
    '''
    Private variable containing the fitness of the individual.

    If the fitness has not yet been calculated, this property will be `None`.
    '''

    def __init__(
        self,
        genome: SatelliteToList[List[Literal[0, 1]]],
        problem_instance: ProblemInstance
    ):
        self.__genome = genome
        self.__problem_instance = problem_instance
        self.__fitness = None

    @property
    def genome(self) -> SatelliteToList[List[Literal[0, 1]]]:
        '''
        The individual's genome.

        This is a property instead of a class variable to ensure it is never
        modified.
        '''
        return self.__genome

    @property
    def problem_instance(self) -> ProblemInstance:
        '''
        The problem instance for all individuals.

        This is a property instead of a class variable to ensure it is never
        modified.
        '''
        return self.__problem_instance

    @property
    def fitness(self) -> float:
        '''
        The fitness of the individual.

        This is a property instead of a class variable so that computations can
        be performed to compute the fitness on-demand.
        '''

        # If the fitness has already been calculated, save some resources and
        # just return it
        if self.__fitness is not None:
            return self.__fitness

        # Increase variance multiplier
        run_genetic_algorithm.variance_multiplier += 1.0 / (POPULATION_SIZE * GENERATIONS)

        # Generate the satellite intervals corresponding to the jobs that are
        # enabled in the individual's genome
        satellite_intervals = get_satellite_intervals_from_genome(
            self.problem_instance,
            self.genome
        )

        # Optimize the schedule using network flow and bin packing algorithms
        optimized_schedule = optimize_schedule(
            self.problem_instance.satellites,
            self.problem_instance.jobs,
            satellite_intervals,
            self.problem_instance.ground_station_passes
        )
        result = optimized_schedule.result

        # Calculate the variance of the amount of jobs scheduled in each
        # satellite
        jobs_in_each_satellite = [
            len(schedule_units)
                for satellite, schedule_units in result.items()
        ]

        # variance = np.var(jobs_in_each_satellite) * total_jobs
        average_jobs_in_each_satellite = np.average(jobs_in_each_satellite)
        variance = 0
        for j in jobs_in_each_satellite:
            variance += abs(average_jobs_in_each_satellite - j)

        # Calculate the weighted sum of the priorities of all jobs
        total_job_priority_sum = sum(
            int(job.priority.value) for job in self.problem_instance.jobs
        )

        # Calculate the weighted sum of the priorities of all jobs that were
        # scheduled
        total_scheduled_job_priority_sum = sum(
            schedule_unit.job.priority.value
                for satellite, schedule_units in result.items()
                    for schedule_unit in schedule_units
        )

        if total_job_priority_sum == 0:
            # Returning 0 here to avoid division by zero exception
            return 0.0

        # Weighted-percentage of jobs scheduled (the weights are the job
        # priorities)
        P = total_scheduled_job_priority_sum

        # Smaller variance is better, so normalize by using 1-variance
        V = variance

        # Fitness formula
        F = P - run_genetic_algorithm.variance_multiplier * V
        logger.debug(f'Fitness: {F}, Percentage: {P}, Variance: {V}, VMultiplier: {run_genetic_algorithm.variance_multiplier}')

        # Set fitness of individual
        self.__fitness = float(F)

        logger.debug(f'Fitness: {self.__fitness}, P: {P}, V: {V}')

        return self.__fitness


@debug
def debug_genetic_info(
    individual: Individual
) -> None:
    '''
    Debugs information about the given individual.

    Args:
        individual: The individual being analyzed.
    '''

    # Directory where debug images will be stored
    DEBUG_DIR = Path.cwd() / 'debug'
    DEBUG_DIR.mkdir(exist_ok=True)

    satellite_intervals = get_satellite_intervals_from_genome(
        individual.problem_instance,
        individual.genome
    )

    # Run network flow algorithm to get debug images
    run_network_flow(
        individual.problem_instance.satellites,
        individual.problem_instance.jobs,
        satellite_intervals,
        DEBUG_DIR
    )


def generate_population(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    outage_requests: List[OutageRequest],
    satellite_intervals: SatelliteToList[SatelliteInterval],
    ground_station_passes: SatelliteToList[GroundStationPassInterval]
) -> Tuple[ProblemInstance, List[Individual]]:
    '''
    Generates an initial population.

    Args:
        satellites: The list of satellites to be scheduled with jobs and outage
            requests.

        jobs: The list of jobs to be scheduled.

        outage_requests: The list of (non-negotiable) outage requests.

        satellite_intervals: The dictionary mapping each satellite to its list
            of schedulable intervals.

        ground_station_passes: The dictionary mapping each satellite to its list
            of ground station passes.

    Returns:
        A tuple containing the problem instance (which has a list of satellites,
        a list of jobs, and a list of outage requests) and a list of individuals
        in the population.
    '''

    # Pack satellites, jobs, and outage requests into problem instance
    problem_instance = ProblemInstance(
        satellites,
        jobs,
        outage_requests,
        satellite_intervals,
        ground_station_passes
    )

    # Generate individuals in the population
    population: List[Individual] = []

    for _ in range(POPULATION_SIZE - 1):
        genome: SatelliteToList[List[Literal[0, 1]]] = {
            sat: [
                [0 if random.random() > INITIAL_ENABLE_PROBABILITY else 1 for job in interval.data]
                for interval in intervals
            ]
            for sat, intervals in satellite_intervals.items()
        }

        individual = Individual(genome, problem_instance)
        population.append(individual)

    # Inject one individual with all 1s (everything enabled)
    genome: SatelliteToList[List[Literal[0, 1]]] = {
        sat: [
            [1 for job in interval.data] for interval in intervals]
            for sat, intervals in satellite_intervals.items()
    }

    individual = Individual(genome, problem_instance)
    population.append(individual)

    return problem_instance, population


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
    problem_instance: ProblemInstance
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

    Returns:
        Either a new individual that is a combination of the two parents, or one
        of the parents, depending (randomly) on the crossover rate.
    '''

    # Randomly choose whether or not to crossover, depending on the crossover
    # rate
    if random.random() < CROSSOVER_RATE:
        new_genome = {sat: [] for sat in problem_instance.satellites}
        for sat in problem_instance.satellites:
            for i in range(len(problem_instance.satellite_intervals[sat])):
                # If either of the parents has negative fitness, just choose the
                # one with the higher fitness (because using `random.choices`
                # does not work with negative weights)
                if parent1.fitness < 0 or parent2.fitness < 0:
                    if parent1.fitness >= parent2.fitness:
                        chosen_parent = parent1
                    else:
                        chosen_parent = parent2
                else:
                    parents = [parent1, parent2]
                    fitnesses = [parent1.fitness, parent2.fitness]

                    # Choose one of the two parents randomly where the parent
                    # with the higher fitness has a higher probability of being
                    # chosen
                    if sum(fitnesses) > 0:
                        chosen_parent = random.choices(
                            parents,
                            weights=fitnesses
                        )[0]
                    else:
                        chosen_parent = random.choice(parents)

                # Use the chromosome from the chosen parent
                chromosome = chosen_parent.genome[sat][i]
                new_genome[sat].append(chromosome)

        return Individual(new_genome, problem_instance)

    # If not crossing over, just return the parent with the higher fitness
    return parent1 if parent1.fitness > parent2.fitness else parent2


def mutate(
    individual: Individual
) -> Individual:
    '''
    Mutate an individual's genome, potentially flipping some of its bits.

    The chance of a bits in a genome being flipped depends on the
    `MUTATION_RATE`.

    Args:
        individual: The individual whose genome is being mutated.

    Returns:
        A new individual with the mutated genome.
    '''

    new_genome: SatelliteToList[List[Literal[0, 1]]] = {
        sat: [
            [
                1 - bit if random.random() > MUTATION_RATE else bit
                    for bit in chromosome
            ]
            for chromosome in chromosomes
        ]
        for sat, chromosomes in individual.genome.items()
    }

    return Individual(
        new_genome,
        individual.problem_instance
    )


def run_genetic_algorithm(
    satellites: List[EarthSatellite],
    jobs: List[Job],
    outage_requests: List[OutageRequest],
    satellite_intervals: SatelliteToList[SatelliteInterval],
    ground_station_passes: SatelliteToList[GroundStationPassInterval]
) -> GeneticAlgorithmResult:
    '''
    The main entry point to the genetic algorithm part of the SOSO scheduling
    algorithm.

    Args:
        satellites: The list of satellites to be scheduled with jobs and outage
            requests.

        jobs: The list of jobs to be scheduled.

        outage_requests: The list of (non-negotiable) outage requests.

        satellite_intervals: The dictionary mapping each satellite to its list
            of schedulable intervals.

        ground_station_passes: The dictionary mapping each satellite to its list
            of ground station passes.

    Returns:
        A dictionary mapping each satellite to a list, where items in the list
        are representations of a job scheduled in a time slot. This is returned
        directly from the network flow algorithm.
    '''

    logger.info('Starting genetic algorithm')

    start_time = time.time()

    run_genetic_algorithm.variance_multiplier = 1
    '''
    Multiplier for the variance that increases over time, so that as the
    algorithm progresses the "equal resource usage constraint" becomes more
    important.
    '''

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
        satellite_intervals,
        ground_station_passes
    )

    for generation in range(GENERATIONS):
        # Make sure all individuals have a fitness metric
        for individual in population:
            if not best_individual:
                # Make sure the best individual is not None
                best_individual = individual
                best_fitness = individual.fitness

        logger.info(f'Generation: {generation}, best fitness: {best_fitness}')
        # Generate a new population by selecting parents, crossing over genomes,
        # and mutating genomes
        new_population: List[Individual] = []
        for _ in range(POPULATION_SIZE):
            parent1 = select(population)
            parent2 = select(population)
            offspring = crossover(parent1, parent2, problem_instance)
            mutated_offspring = mutate(offspring)
            new_population.append(mutated_offspring)

        # Replace the original population with the new population
        population = new_population

        # Update the best individual with the new population
        for individual in population:
            if individual.fitness > best_fitness:
                best_individual = individual
                best_fitness = individual.fitness

        if best_individual:
            debug_genetic_info(best_individual)

    if not best_individual:
        raise Exception('Genetic algorithm did not find a best individual')

    end_time = time.time()

    logger.info(f'Genetic algorithm took {end_time-start_time} seconds')

    best_satellite_intervals = get_satellite_intervals_from_genome(
        problem_instance,
        best_individual.genome
    )

    # Run the network flow algorithm one last time to get the solution to the
    # problem instance and the best individual's genome
    return optimize_schedule(
        problem_instance.satellites,
        problem_instance.jobs,
        best_satellite_intervals,
        problem_instance.ground_station_passes
    )
