from dataclasses import dataclass

from ortools.linear_solver import pywraplp


@dataclass(frozen=True)
class Item:
    name: str
    size: int


@dataclass(frozen=True)
class Bin:
    name: str
    capacity: int


# Data
items = [
    Item('A', 1000),
    Item('B', 256),
    Item('C', 512),
    Item('D', 512),
    Item('E', 256),
    Item('F', 128)
]
bin_capacities = [
    Bin('X', 1000),
    Bin('Y', 700),
    Bin('Z', 290),
    Bin('wildcard', 1000),
]

num_bins = len(bin_capacities)
num_items = len(items)

solver = pywraplp.Solver.CreateSolver("SCIP")

# Decision variables: x[i][j] = 1 if item i is in bin j
x = {}
for i in range(len(items)):
    for j in range(num_bins):
        x[i, j] = solver.BoolVar(f"x[{i},{j}]")

# y[j] = 1 if bin j is used
y = [solver.BoolVar(f"y[{j}]") for j in range(num_bins)]

# Constraints: each item must be assigned to at most one bin
for i in range(len(items)):
    solver.Add(sum(x[i, j] for j in range(num_bins)) <= 1)

# Constraints: each bin cannot hold more than its capacity
for j in range(num_bins):
    solver.Add(
        sum(
            items[i].size * x[i, j] for i in range(len(items))
        ) <= bin_capacities[j].capacity * y[j]
    )

# Item 0 (Item A) cannot be placed in bin 1 (Bin X)
solver.Add(x[0, 1] == 0)

# Objective: Minimize the number of bins used
# solver.Minimize(solver.Sum(y))

# Objective: maximize the number of items packed
solver.Maximize(sum(x[i, j] for i in range(num_items) for j in range(num_bins)))

bin_contents = {bin: [] for bin in bin_capacities}
unpacked_items = []

# Solve
status = solver.Solve()
if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    if status == pywraplp.Solver.OPTIMAL:
        print(f'Optimal solution\n')
    elif status == pywraplp.Solver.FEASIBLE:
        print(f'Feasible solution\n')

    for i in range(len(items)):
        packed = False
        for j in range(num_bins):
            if x[i, j].solution_value() > 0:
                bin_contents[bin_capacities[j]].append(items[i])
                packed = True

        if not packed:
            unpacked_items.append(items[i])

    import pprint
    print("Packed bins:")
    print('------------')
    pprint.pp(bin_contents)
    print("\nUnpacked items:")
    print('-----------------')
    pprint.pp(unpacked_items)

else:
    print("No optimal or feasible solution found")


import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, List

def visualize_packing(bin_contents: Dict[Bin, List[Item]], bins, unpacked_items):
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {}  # Store colors for each item
    bin_labels = []
    y_offsets = np.zeros(len(bins))  # Track height for stacking items in each bin

    for bin_idx, (bin, items) in enumerate(bin_contents.items()):
        bin_labels.append(bin.name)
        for item in items:
            # Assign a random color to each item (or reuse existing)
            if item not in colors:
                colors[item] = (
                    random.random(),
                    random.random(),
                    random.random()
                )

            # item_size = next(obj.size for obj in items if obj.name == item)

            ax.bar(
                bin_idx,
                item.size,
                bottom=y_offsets[bin_idx],
                color=colors[item],
                edgecolor='black',
                label=item.name if item not in colors else ""
            )
            y_offsets[bin_idx] += item.size  # Stack items upwards

            # Label items inside bins
            ax.text(
                bin_idx,
                y_offsets[bin_idx] - item.size / 2,
                item.name,
                ha='center',
                va='center',
                fontsize=10,
                color='white',
                weight='bold'
            )

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("Bin Capacity Used")
    ax.set_title("Bin Packing Visualization")
    ax.legend(loc="upper right", title="Items", bbox_to_anchor=(1.2, 1))
    
    # Show unpacked items if any
    if unpacked_items:
        plt.figtext(0.9, 0.02, f"⚠ Unpacked Items: {', '.join(unpacked_items)}", fontsize=10, color='red', ha='right')

    plt.show()

# Call the visualization function after solving
visualize_packing(bin_contents, bin_capacities, unpacked_items)
