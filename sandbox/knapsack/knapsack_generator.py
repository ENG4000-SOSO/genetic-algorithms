import random
from typing import Tuple

def generate_knapsack(n: int = 1_000_000) -> Tuple[int, int]:
    weights = [0] * n
    values = [0] * n
    for i in range(n):
        randomWeight = random.randint(1, 10)
        weights[i] = randomWeight

        randomValue = random.randint(1, 10)
        values[i] = randomValue

    return weights, values
