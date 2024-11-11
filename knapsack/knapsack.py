import time
from ortools.algorithms.python import knapsack_solver
from knapsack_generator import generate_knapsack
from genetic_knapsack import genetic_algorithm


def knapsack_brute_force(weights, values, capacity, n):
    # Base case: no items or no capacity left
    if n == 0 or capacity == 0:
        return 0

    # If the weight of the nth item is more than the current capacity
    # We can't include this item in the knapsack
    if weights[n-1] > capacity:
        return knapsack_brute_force(weights, values, capacity, n-1)

    # Otherwise, we have two choices:
    # 1. Include the item in the knapsack
    # 2. Exclude the item from the knapsack
    # We return the maximum of the two
    else:
        include_item = values[n-1] + knapsack_brute_force(weights, values, capacity - weights[n-1], n-1)
        exclude_item = knapsack_brute_force(weights, values, capacity, n-1)
        return max(include_item, exclude_item)


def knapsack_dp(weights, values, capacity, n):
    # Create a 2D DP array with (n+1) rows and (capacity+1) columns
    dp = [[0 for x in range(capacity + 1)] for x in range(n + 1)]

    # Build the DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                # Either include the item or exclude it
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                # Exclude the item if it exceeds current weight limit
                dp[i][w] = dp[i-1][w]

    # The answer is in the bottom-right cell
    return dp[n][capacity]


# # Example usage:
# weights = [2, 3, 4, 5]
# values = [3, 4, 5, 6]
# capacity = 5
# n = len(values)

n = 1_000
capacity = n // 5
weights, values = generate_knapsack(n)

# t0 = time.time()
# bruteForceSolution = knapsack_brute_force(weights, values, capacity, n)
# t1 = time.time()
# print(f"Brute force solution: {bruteForceSolution} (took {t1 - t0} seconds)")

t0 = time.time()
dynamicProgrammingSolution = knapsack_dp(weights, values, capacity, n)
t1 = time.time()
print(f"Dynamic programming solution: {dynamicProgrammingSolution} (took {t1 - t0} seconds)")

# solver = knapsack_solver.KnapsackSolver(
#     knapsack_solver.SolverType.KNAPSACK_BRUTE_FORCE_SOLVER,
#     "example1"
# )
# solver.init(values, [weights], [capacity])
# t0 = time.time()
# orToolsSolution = solver.solve()
# t1 = time.time()
# print(f"OR Tools brute force solution: {orToolsSolution} (took {t1 - t0} seconds)")

solver = knapsack_solver.KnapsackSolver(
    knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
    "example2"
)
solver.init(values, [weights], [capacity])
t0 = time.time()
orToolsSolution = solver.solve()
t1 = time.time()
print(f"OR Tools branch and bound solution: {orToolsSolution} (took {t1 - t0} seconds)")

solver = knapsack_solver.KnapsackSolver(
    knapsack_solver.SolverType.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
    "example3"
)
solver.init(values, [weights], [capacity])
t0 = time.time()
orToolsSolution = solver.solve()
t1 = time.time()
print(f"OR Tools dynamic programming solution: {orToolsSolution} (took {t1 - t0} seconds)")

t0 = time.time()
_ ,geneticAlgorithmSolution = genetic_algorithm(weights, values, capacity)
t1 = time.time()
print(f"Genetic algorithm solution: {geneticAlgorithmSolution} (took {t1 - t0} seconds)")
