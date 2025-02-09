from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary

# Data
items = [128, 256, 512, 512, 256, 128]
# bin_capacities = [700, 1000, 200]
bin_capacities = [700, 1000, 290]
num_bins = len(bin_capacities)

# LP Model
model = LpProblem("BinPacking", LpMinimize)

# Variables
x = [[LpVariable(f"x_{i}_{j}", cat=LpBinary) for j in range(num_bins)] for i in range(len(items))]
y = [LpVariable(f"y_{j}", cat=LpBinary) for j in range(num_bins)]

# Objective: Minimize the number of bins used
model += lpSum(y)

# Constraints: Each item must be in exactly one bin
for i in range(len(items)):
    model += lpSum(x[i][j] for j in range(num_bins)) == 1

# Constraints: Bin capacities (each bin has a different capacity)
for j in range(num_bins):
    model += lpSum(items[i] * x[i][j] for i in range(len(items))) <= bin_capacities[j] * y[j]

# Solve
model.solve()

# Print results
bins = [[] for _ in range(num_bins)]
for i in range(len(items)):
    for j in range(num_bins):
        if x[i][j].varValue == 1:
            bins[j].append(items[i])

bins = [b for b in bins if b]
print("Packed bins:", bins)
